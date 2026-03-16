import tensorrt as trt
import os
import sys
import numpy as np
import logging
import argparse
import glob
from pathlib import Path
from model_handler import MODEL_REGISTRY, get_default_onnx_dir, download_model
from cuda_utilities import Common  # Import Common class for CUDA utilities
class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose= False, workspace=12):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.network = None
        self.parser = None

    def create_network(self, onnx_model_idx):
        """
        Parse the ONNX graph and create the TensorRT network definition.
        Adds dynamic shape support for batch size [1..32].
        """
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        
        onnx_path = download_model(onnx_model_idx,"models")
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print(f"ERROR: Failed to load ONNX file {onnx_path}")
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")
        # Define optimization profile for dynamic batch
        profile = self.builder.create_optimization_profile()
        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            print(f"Input: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

            # Replace -1 with dynamic range
            min_shape = (1, 256, 256, 1)
            opt_shape = (8, 256, 256, 1)
            max_shape = (32, 256, 256, 1)
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)

        self.config.add_optimization_profile(profile)

    def create_engine_fp(self, input_name,precision="fp32"):
        input_name = Path(input_name)
        stem = input_name.stem if input_name.suffix == ".engine" else input_name.name
        engine_name = f"{stem}-{precision}.engine"
        engine_path = Path("engines") / engine_name
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        if precision=="fp16" and not self.builder.platform_has_fast_fp16:
            print("WARNING: FP16 not natively supported on this platform.")
        if precision=="fp16":
            self.config.set_flag(trt.BuilderFlag.FP16)
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            raise RuntimeError("Failed to create engine.")

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        if precision=="fp16":
            print(f"FP16 Engine saved at: {engine_path}")
        elif precision=="fp32":
            print(f"FP32 Engine saved at: {engine_path}")

def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx)
    builder.create_engine_fp(args.engine,args.precision)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--onnx", required=True, help="The input ONNX model ('a' - 'k') file to load"
    )
    parser.add_argument(
        "-e", "--engine", required=True, help="The output name for the TRT engine ;engine is stored '[Fetch_V2]/engines/'"
    )
    parser.add_argument(
        "-p",
        "--precision",
        default="fp32",
        choices=["fp32", "fp16"],
        help="The precision mode to build in, either fp32/fp16, default: fp32",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable more verbose log output"
    )
    parser.add_argument(
        "-w",
        "--workspace",
        default=8,
        type=int,
        help="The max memory workspace size to allow in Gb, default: 8",
    )
    args = parser.parse_args()
    main(args)


'''
# ==== Colab usage unit testing hardcoded ====
onnx_file = "/content/model_a.onnx"
engine_file = "model-a-fp16.engine"

builder = EngineBuilder(verbose=True)
builder.create_network(onnx_file)  # Now supports dynamic batch [1..32]
builder.create_engine_fp16(engine_file)

# ==== Terminal usage====
python build_engine.py \
  --onnx a \
  --engine model_a.engine \
  --precision fp32 \
  --workspace 8 \
  --verbose
```

**What each arg maps to in your code:**

| Arg | Maps to | Effect |
|---|---|---|
| `--onnx a` | `builder.create_network("a")` | downloads/uses `model_a.onnx` from registry |
| `--engine model_a.engine` | `create_engine_fp("model_a.engine")` | saves to `engines/model_a.engine` |
| `--precision fp16` | `config.set_flag(FP16)` | builds FP16 engine |
| `--workspace 8` | `set_memory_pool_limit(8 * 2^30)` | 8 GB builder RAM |
| `--verbose` | `trt.Logger.VERBOSE` | full TRT build logs |

**Expected output if it works:**
```
Input: data_freq_time, shape: (-1, 256, 256, 1), dtype: DataType.FLOAT
Input: data_dm_time,   shape: (-1, 256, 256, 1), dtype: DataType.FLOAT
FP32 Engine saved at: engines/model_a.engine

'''