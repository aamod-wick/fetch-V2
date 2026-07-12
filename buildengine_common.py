"""
TensorRT INT8 Engine Builder for FETCH Fast Radio Burst Detection.
Calibration uses H5 files containing freq-time and DM-time data.
"""

from pathlib import Path

import tensorrt as trt
import os
import sys
import numpy as np
import argparse
import glob
import scipy.signal as s
import h5py
from cuda import cudart
from cuda_utilities import Common
from model_handler import download_model 
from data_handler import  h5_batch_generator 


# ---------------------------------------------------------------------------
# INT8 Calibrator
# ---------------------------------------------------------------------------

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Entropy Calibrator using H5 files as calibration data.

    The model has two inputs:
      - freq-time (ft): shape (B, 256, 256, 1)
      - DM-time   (dt): shape (B, 256, 256, 1)

    get_batch() returns [ft_gpu_ptr, dt_gpu_ptr].
    The order of pointers matches the order TensorRT passes in `names`,
    which reflects the ONNX input order. Verify with a debug print if unsure.
    """

    def __init__(self, cache_file, h5_files, calib_batch_size=8,dm_time_only = False):
        super().__init__()
        self.cache_file = cache_file
        self.calib_batch_size = calib_batch_size
        self.total = len(h5_files)
        self.processed = 0
        self.dm_time_only = dm_time_only
        self.common = Common()

        # GPU allocations for both inputs — fixed shape (B, 256, 256, 1) float32
        size = int(np.dtype(np.float32).itemsize * calib_batch_size * 256 * 256 * 1)
        if(not dm_time_only):
            self.ft_allocation = self.common.cuda_call(cudart.cudaMalloc(size))
        self.dt_allocation = self.common.cuda_call(cudart.cudaMalloc(size))

        # Wire up the generator only if there are files to process
        if h5_files:
            self.batch_generator = h5_batch_generator(h5_files, batch_size=calib_batch_size, dm_time_only=dm_time_only)
        else:
            self.batch_generator = iter([])  # empty — will rely on cache

    def get_batch_size(self):
        return self.calib_batch_size

    def get_batch(self, names):
        """
        Called repeatedly by TensorRT until None is returned.
        `names` contains the ONNX input names in TensorRT's expected order.
        Returned pointer list must match that same order.
        """
        try:
            if(not self.dm_time_only):
                ft_batch, dt_batch, files = next(self.batch_generator)
                self.common.memcpy_host_to_device(self.ft_allocation, np.ascontiguousarray(ft_batch))#only account ft when not using dm_time_only moddel for calibration
            else:
                dt_batch, files = next(self.batch_generator)
            self.common.memcpy_host_to_device(self.dt_allocation, np.ascontiguousarray(dt_batch))
            self.processed += len(files) #just a manual check to see how many files are processed 
            print(f"[CALIBRATION] Processed {self.processed} / {self.total} files")

            if(not self.dm_time_only):
                # Order: ft first, dt second — must match ONNX export input order
                return [int(self.ft_allocation), int(self.dt_allocation)]
            else:
                # Only DM-time input for calibration
                return [int(self.dt_allocation)]
        except StopIteration:
            print("[CALIBRATION] All calibration batches complete.")
            return None

    def read_calibration_cache(self):
        if self.cache_file is not None and os.path.exists(self.cache_file):
            print(f"[CALIBRATION] Loading cache from: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        if self.cache_file is None:
            return
        print(f"[CALIBRATION] Writing cache to: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ---------------------------------------------------------------------------
# Engine Builder — INT8 only
# ---------------------------------------------------------------------------

class EngineBuilder:
    """
    Parses an ONNX graph and builds a engine : supported precisions : FP32, FP16, INT8,FP 8.
    """

    def __init__(self, verbose=False, workspace=8, dm_time_only=False, precision="FP32"):
        self.trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30)
        )

        self.network = None
        self.parser = None
        self.dm_time_only = dm_time_only
        self.precision = precision.lower()

    def create_network(self, onnx_model_id, dynamic_batch_size=None, local=False):
        """
        Parse ONNX and create the TensorRT network.

        :param onnx_model_id: ID of the ONNX model to download and parse.
        :param dynamic_batch_size: Comma-separated MIN,OPT,MAX or list of 3 ints.
                                   OPT is a tuning hint and is independent of calib batch size.
        """
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        if(not local):
            onnx_path = download_model(onnx_model_id, "models")
        else:
            onnx_path = Path(onnx_model_id)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print(f"[ERROR] Failed to parse ONNX: {onnx_path}")
                for i in range(self.parser.num_errors):
                    print(f"[ERROR] {self.parser.get_error(i)}")
                sys.exit(1)

        print("\n--- Network Inputs ---")
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        profile = self.builder.create_optimization_profile()
        has_dynamic = False

        for inp in inputs:
            print(f"  '{inp.name}'  shape={inp.shape}  dtype={inp.dtype}")
            if inp.shape[0] == -1:
                has_dynamic = True
                if dynamic_batch_size is not None:
                    if isinstance(dynamic_batch_size, str):
                        dynamic_batch_size = [int(v) for v in dynamic_batch_size.split(",")]
                    assert len(dynamic_batch_size) == 3, "dynamic_batch_size must be MIN,OPT,MAX"
                    b_min, b_opt, b_max = dynamic_batch_size
                    min_shape = (b_min, 256, 256, 1)
                    opt_shape = (b_opt, 256, 256, 1)
                    max_shape = (b_max, 256, 256, 1)
                else:
                    # dyanamic fallback if params not spcifieid, default to 1,8,32
                    min_shape = (1, 256, 256, 1)
                    opt_shape = (8, 256, 256, 1)
                    max_shape = (32, 256, 256, 1)
                profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
                print(f"  Profile: MIN={min_shape}  OPT={opt_shape}  MAX={max_shape}")

        if has_dynamic:
            self.config.add_optimization_profile(profile)

        print("\n--- Network Outputs ---")
        for i in range(self.network.num_outputs):
            out = self.network.get_output(i)
            print(f"  '{out.name}'  shape={out.shape}  dtype={out.dtype}")
        print()

    def create_engine(
        self,
        input_name,
        calib_input=None,
        calib_cache=None,
        calib_num_images=None,
        calib_batch_size=None,
    ):
        """
        Build and serialize the TensorRT engine in precisions : FP32, FP16, INT8, FP8.

        :param input_name: Name of the engine.
        :param calib_input: Directory containing H5 files for calibration.
        :param calib_cache: Path to read/write the INT8 calibration cache.
        :param calib_num_images: Max number of H5 files to use for calibration.
        :param calib_batch_size: Samples per calibration forward pass.
        """
        input_name = Path(input_name)
        stem = input_name.stem if input_name.suffix == ".engine" else input_name.name
        engine_name = f"{stem}-{self.precision.upper()}.engine"
        engine_path = Path("engines") / engine_name
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        precision = self.precision
        if precision == "int8" and not self.builder.platform_has_fast_int8:
            print("[WARNING] INT8 is not natively supported on this device — may fall back to FP32.")
        if precision == "fp16" and not self.builder.platform_has_fast_fp16:
            print("[WARNING] FP16 is not natively supported on this device — may fall back to FP32.")
        self.config.set_flag(self.precision.upper())

        # --- Set up calibrator ---
        if precision == "int8":
            if calib_cache is not None and os.path.exists(calib_cache):
                print(f"[INFO] Existing calibration cache found at {calib_cache} — skipping H5 calibration.")
                h5_files = []
            else:
                if calib_input is None:
                    print("[ERROR] calib_input directory required when no calibration cache exists.")
                    sys.exit(1)
                h5_files = sorted(glob.glob(os.path.join(calib_input, "**/*.h5"), recursive=True))
                if not h5_files:
                    print(f"[ERROR] No H5 files found in: {calib_input}")
                    sys.exit(1)
                h5_files = h5_files[:calib_num_images]
                print(f"[INFO] Using {len(h5_files)} H5 files for INT8 calibration (batch_size={calib_batch_size}).")

            self.config.int8_calibrator = EngineCalibrator(cache_file=calib_cache,h5_files=h5_files,calib_batch_size=calib_batch_size)       
        # --- Build ---
        print(f"[INFO] Building {precision.upper()} engine -> {engine_path}")
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            print("[ERROR] Engine build failed.")
            sys.exit(1)

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"[INFO] Engine serialized to: {engine_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main(args):
    builder = EngineBuilder(verbose=args.verbose, workspace=args.workspace,precision = args.precision,dm_time_only=args.dm_time_only)
    builder.create_network(onnx_model_id=args.onnx,dynamic_batch_size=args.dynamic_batch_size)
    builder.create_engine(input_name=args.engine,calib_input=args.calib_input,calib_cache=args.calib_cache,calib_num_images=args.calib_num_images,calib_batch_size=args.calib_batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a TensorRT INT8 engine for FETCH FRB detection."
    )
    parser.add_argument("-o", "--onnx", required=True,
                        help="INDEX of input ONNX model")
    parser.add_argument("-e", "--engine", required=True,
                        help="Output name for the TRT engine")
    parser.add_argument("-d", "--dynamic_batch_size", default=[1, 8, 32], type=str,
                        help="Dynamic batch size as MIN,OPT,MAX e.g. 1,8,32. "
                             "OPT is a tuning hint and is independent of calib_batch_size.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose TensorRT logging")
    parser.add_argument("-w", "--workspace", default=8, type=int,
                        help="Max GPU workspace in GB, default: 8")
    parser.add_argument("--calib_input", default=None,
                        help="Directory of H5 files for INT8 calibration")
    parser.add_argument("--calib_cache", default=None,
                        help="Path to read/write calibration cache")
    parser.add_argument("--calib_num_images", default=500, type=int,
                        help="Max H5 files to use for calibration, default: 500")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="Batch size per calibration pass, default: 8")
    parser.add_argument("--precision", default="int8", choices=["int8", "fp16", "fp32"],
                        help="Precision mode to build in, default: int8")
    parser.add_argument("-D","--dm_time_only",default=False,type=bool,help="Only use DM-Time data for inference")

    parser.add_argument("--local", default=False,type=bool, help="Use local(custom) models instead of downloading from the model zoo")

    args = parser.parse_args()

    if args.calib_cache is None or not os.path.exists(args.calib_cache):
        if args.calib_input is None:
            parser.print_help()
            print("\n[ERROR] Provide --calib_input (H5 directory) or an existing --calib_cache.")
            sys.exit(1)

    main(args)