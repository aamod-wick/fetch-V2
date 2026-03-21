from time import time

import tensorrt as trt
import numpy as np
from cuda import cudart
from pathlib import Path
import argparse
import glob
import pandas as pd 
import os
from cuda_utilities import Common  # Import Common class
from data_handler import load_and_preprocess_h5_data, process_batch,find_dm_of_file,sort_h5_files_by_dm #import data handling functions for H5 files

class TensorRTInfer:
    """
    Implements inference for a two-input TensorRT engine with dynamic batching.
    """
    

    def __init__(self, engine_path,common=None):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        if common is None:
            common = Common()
        self.common = common
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.inputs_by_name = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))

            # For dynamic shapes, we don't set a fixed shape during initialization
            # We'll set the shape during inference based on actual input
            shape = self.context.get_tensor_shape(name)

            # Store the tensor info without fixed allocation
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "is_input": is_input,
            }

            if is_input:
                self.inputs.append(binding)
                self.inputs_by_name[name] = binding
            else:
                self.outputs.append(binding)

            print(
                "{} '{}' with dynamic shape and dtype {}".format(
                    "Input" if is_input else "Output",
                    name,
                    dtype,
                )
            )

    def set_input_shapes(self, input_shapes):
        """
        Set the input shapes for dynamic inference.
        :param input_shapes: Dictionary with input names as keys and shapes as values
        """
        for name, shape in input_shapes.items():
            if not self.context.set_input_shape(name, shape):
                raise ValueError(f"Failed to set shape {shape} for input {name}")

        # Update output shapes and allocate memory
        self._allocate_io_buffers()

    def _allocate_io_buffers(self):
        """Allocate I/O buffers based on current context shapes"""
        # Clear previous allocations
        for allocation in self.allocations:
            self.common.cuda_call(cudart.cudaFree(allocation))
        self.allocations.clear()

        # Allocate for all tensors
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)

            # Calculate memory size and allocate
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = self.common.cuda_call(cudart.cudaMalloc(size))

            # Update binding information
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                binding = self.inputs_by_name[name]
            else:
                # Find output binding
                binding = next((out for out in self.outputs if out["name"] == name), None)

            if binding:
                binding["shape"] = list(shape)
                binding["allocation"] = allocation
                if not is_input:
                    binding["host_allocation"] = np.zeros(shape, dtype)

            self.allocations.append(allocation)

    def infer(self, ft_batch, dt_batch):
        """
        Execute inference on FT and DT batches with dynamic shapes.
        """
        # Set input shapes based on actual data
        input_shapes = {
            "data_freq_time": ft_batch.shape,
            "data_dm_time": dt_batch.shape
        }
        self.set_input_shapes(input_shapes)

        # Copy inputs to device
        ft_input = self.inputs_by_name.get("data_freq_time")
        dt_input = self.inputs_by_name.get("data_dm_time")

        if ft_input is None or dt_input is None:
            # Try alternative names
            ft_input = self.inputs_by_name.get("ft_batch")
            dt_input = self.inputs_by_name.get("dt_batch")

        if ft_input is None or dt_input is None:
            raise ValueError("Could not find input bindings for FT and DT data")

        self.common.memcpy_host_to_device(ft_input["allocation"], ft_batch.ravel())
        self.common.memcpy_host_to_device(dt_input["allocation"], dt_batch.ravel())

        # Execute inference
        self.context.execute_v2(self.allocations)

        # Copy outputs from device to host
        outputs = {}
        for output in self.outputs:
            host_array = np.zeros(output["shape"], dtype=output["dtype"])
            self.common.memcpy_device_to_host(host_array, output["allocation"])
            outputs[output["name"]] = host_array

        return outputs

    def input_spec(self):
        """Get current input specifications"""
        specs = {}
        for inp in self.inputs:
            shape = self.context.get_tensor_shape(inp["name"])
            specs[inp["name"]] = (list(shape), inp["dtype"])
        return specs
# ─────────────────────────────────────────────
#  H5 inference pipeline
# ─────────────────────────────────────────────

def resolve_engine_path(engine_name: str, suffix: str = ".engine") -> Path:
    """
    Build engine path as:  engines/<engine_name><suffix>
    Creates the engines/ directory if it doesn't exist.
    """
    engine_path = Path("engines") / (engine_name + suffix)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    return engine_path


def run_inference_on_h5_folder(engine_path: Path, h5_folder: Path,
                                batch_size: int = 8,
                                ft_dim: tuple = (256, 256),
                                dt_dim: tuple = (256, 256)):
    """
    Run TensorRT inference on all H5 files in a folder.

    :param engine_path: Resolved Path to the .engine file
    :param h5_folder:   Path to folder containing .h5 candidate files
    :param batch_size:  Number of files to process per batch
    :param ft_dim:      Freq-time spatial dims expected by the model
    :param dt_dim:      DM-time spatial dims expected by the model
    :return:            Dict mapping filename → model output arrays
    """
    h5_files = sorted(glob.glob(str(h5_folder / "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {h5_folder}")

    print(f"Found {len(h5_files)} H5 files in {h5_folder}")
    print(f"Loading engine from {engine_path}")

    inferrer = TensorRTInfer(engine_path)
    all_results = {}

    for ft_batch, dt_batch, batch_files in process_batch(
        h5_files, batch_size=batch_size, ft_dim=ft_dim, dt_dim=dt_dim
    ):
        # Model expects (N, H, W, 1) — process_batch already adds channel dim
        print(f"Running inference on batch of {len(batch_files)} files")
        outputs = inferrer.infer(ft_batch, dt_batch)

        # Map each file to its corresponding row in the output arrays
        for idx, h5_file in enumerate(batch_files):
            all_results[Path(h5_file).name] = {
                key: val[idx] for key, val in outputs.items()
            }

    return all_results
def run_timed_inference_on_h5_folder(engine_path: Path, h5_folder: Path,DM_value: float, batch_size: int = 8, ft_dim: tuple = (256, 256), dt_dim: tuple = (256, 256), repetitions: int = 10, timing_result_path: str = "timing_results.csv"):
    """Run inference multiple times to measure latency and save results to CSV
    Get the timing results for each run take the average of the runs and save ot to CSV
    Format of CSV is {dm_value, latency_sec}
    
    :param engine_path: Resolved Path to the .engine file
    :param h5_folder:   Path to folder containing .h5 candidate files (are sorted into subfolders by DM value)
    :param DM_value:    The DM value for the current batch
    :param batch_size:  Number of files to process per batch
    :param ft_dim:      Freq-time spatial dims expected by the model
    :param dt_dim:      DM-time spatial dims expected by the model
    :param repetitions: Number of times to repeat inference for timing
    :param timing_result_path: Path to save timing results CSV
    """
    timing_results = []
    for i in range(repetitions):
        start_time = time()
        results = run_inference_on_h5_folder(engine_path, h5_folder, batch_size, ft_dim, dt_dim)
        end_time = time()
        latency = end_time - start_time
        timing_results.append(latency)
        print(f"Run {i+1}/{repetitions}: Latency = {latency:.4f} seconds")
    
    # Calculate average and save single result to CSV
    avg_latency = np.mean(timing_results)
    
    summary_df = pd.DataFrame([{
        "DM_value": DM_value,
        "avg_latency_sec": avg_latency,
    }])
    
    summary_df.to_csv(timing_result_path, index=False)
    print(f"\nTiming summary saved to: {timing_result_path}")
    print(f"Average latency: {avg_latency:.4f}")
    
    return results

def main(args):
    engine_path = resolve_engine_path(args.engine_name, suffix=args.engine_suffix)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    h5_folder = Path(args.h5_folder)
    if not h5_folder.is_dir():
        raise NotADirectoryError(f"H5 folder not found: {h5_folder}")
    if args.run_timing:
        # Iterate first-level subdirectories only (each subdir = one DM bucket)
        dm_subdirs = sorted([p for p in h5_folder.iterdir() if p.is_dir()])
        if not dm_subdirs:
            raise FileNotFoundError(f"No subdirectories found in {h5_folder}")

        for subdir in dm_subdirs:
            dm_value = find_dm_of_file(str(subdir))
            print(f"\nRunning timed inference for {subdir.name} (DM={dm_value})")
            run_timed_inference_on_h5_folder(
                engine_path=engine_path,
                h5_folder=subdir,  # run on subdir, not main folder
                DM_value=dm_value,
                batch_size=args.batch_size,
                ft_dim=tuple(args.ft_dim),
                dt_dim=tuple(args.dt_dim),
                repetitions=args.timing_repetitions,
                timing_result_path=args.timing_result_path,
            )
        print(f"\nInference complete — {len(dm_subdirs)} DM buckets processed.")
    else:
        results = run_inference_on_h5_folder(
        engine_path=engine_path,
        h5_folder=h5_folder,
        batch_size=args.batch_size,
        ft_dim=tuple(args.ft_dim),
        dt_dim=tuple(args.dt_dim),
    )

        print(f"\nInference complete — {len(results)} candidates processed.")
    for fname, output in results.items():
        for out_name, arr in output.items():
            print(f"  {fname}  |  {out_name}: {arr}")
    all_candidates = []
    all_probabilities = []

    for fname, output in results.items():
        # output is a dict of {output_name: array} per file
        # assumes first output contains [prob_negative, prob_positive]
        first_output = next(iter(output.values()))
        prob = float(first_output[1])          # column 1 = FRB probability
        all_candidates.append(fname)
        all_probabilities.append(prob)

    results_dict = {
        "candidate":   all_candidates,
        "probability": all_probabilities,
        "label": (np.array(all_probabilities) >= args.probability).astype(int),
    }
    if(args.results_file):
        results_file = Path(args.results_file)
    else:
        results_file = Path(args.h5_folder) / f"results_{args.engine_name}_trt.csv"
    pd.DataFrame(results_dict).to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

    num_detections = sum(results_dict["label"])
    print(f"{len(all_candidates)} candidates processed, "
          f"{num_detections} detections above threshold {args.probability}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT inference on H5 candidate files")

    parser.add_argument(
        "--engine_name", type=str, required=True,
        help="Engine name without extension, e.g. 'model_a' → engines/model_a.engine"
    )
    parser.add_argument(
        "--engine_suffix", type=str, default=".engine",
        help="Engine file extension (default: .engine)"
    )
    parser.add_argument(
        "--h5_folder", type=str, required=True,
        help="Path to folder containing .h5 candidate files"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Number of H5 files per inference batch (default: 8)"
    )
    parser.add_argument(
        "--ft_dim", type=int, nargs=2, default=[256, 256],
        metavar=("H", "W"),
        help="Freq-time spatial dimensions (default: 256 256)"
    )
    parser.add_argument(
        "--dt_dim", type=int, nargs=2, default=[256, 256],
        metavar=("H", "W"),
        help="DM-time spatial dimensions (default: 256 256)"
    )
    parser.add_argument(
    "--probability", type=float, default=0.5,
    help="Detection probability threshold (default: 0.5)"
    )
    parser.add_argument(
    "--results_file", type=str, default=None,
    help="Optional path to save results CSV (default: <h5_folder>/results_<engine_name>_trt.csv)"
    )
    parser.add_argument(
        "--timing_result_path", type=str, default="timing_results.csv",
        help="Path to save timing results CSV (default: timing_results.csv)"
    )
    parser.add_argument(
        "--timing_repetitions", type=int, default=10,
        help="Number of repetitions for timing inference (default: 10)"
    )
    parser.add_argument(
        "--run_timing", type =bool, default=False,
        help="If set, runs timed inference and saves timing results to CSV"
    )

    args = parser.parse_args()
    main(args)
#example usage:
'''
python trt_infer.py \
  --engine_name model_a \
  --h5_folder /content/candidates \
  --batch_size 4 \
  --probability 0.5 \
  --ft_dim 256 256 \
  --dt_dim 256 256 \
  --engine_suffix .engine
```

**What each arg maps to in your code:**

| Arg | Maps to | Effect |
|---|---|---|
| `--engine_name model_a` | `resolve_engine_path("model_a", ".engine")` | looks for `engines/model_a.engine` |
| `--h5_folder /content/candidates` | `Path(args.h5_folder)` | folder scanned for `*.h5` files |
| `--batch_size 4` | passed to `process_batch()` | 4 h5 files per GPU batch |
| `--probability 0.5` | `>= args.probability` threshold | label = 1 if score ≥ 0.5 |
| `--engine_suffix .engine` | appended to engine name | can change to `.trt` if needed |

**Output you should expect if it works:**
```
Found N H5 files in /content/candidates
Loading engine from engines/model_a.engine
Running inference on batch of 4 files
...
Inference complete — N candidates processed.
Results saved to: /content/candidates/results_model_a_trt.csv
N candidates processed, X detections above threshold 0.5




'''