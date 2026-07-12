"""
TensorRT INT8 Engine Builder for FETCH Fast Radio Burst Detection.
Calibration uses H5 files containing freq-time and DM-time data.
Supports FP32, FP16, INT8, FP8, FP4 precisions.
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
from data_handler import h5_batch_generator


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

    def __init__(self, cache_file, h5_files, calib_batch_size=8, dm_time_only=False):
        super().__init__()
        self.cache_file = cache_file
        self.calib_batch_size = calib_batch_size
        self.total = len(h5_files)
        self.processed = 0
        self.dm_time_only = dm_time_only
        self.common = Common()
        self.batch_generator = None
        self.ft_allocation = None
        self.dt_allocation = None

        # GPU allocations for inputs — fixed shape (B, 256, 256, 1) float32
        size = int(np.dtype(np.float32).itemsize * calib_batch_size * 256 * 256 * 1)
        
        # Always allocate DT memory
        self.dt_allocation = self.common.cuda_call(cudart.cudaMalloc(size))
        print(f"[DEBUG] Allocated DT memory: {size} bytes")
        
        # Only allocate FT memory if needed
        if not dm_time_only:
            self.ft_allocation = self.common.cuda_call(cudart.cudaMalloc(size))
            print(f"[DEBUG] Allocated FT memory: {size} bytes")
        else:
            self.ft_allocation = None

        # Wire up the generator only if there are files to process
        if h5_files:
            print(f"[DEBUG] First 5 H5 files: {h5_files[:5]}")
            self.batch_generator = h5_batch_generator(
                h5_files, 
                batch_size=calib_batch_size,
                dm_time_only=dm_time_only
            )
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
        # DEBUG: Print names to see the order TensorRT expects
        print(f"[DEBUG] Calibration input names from TRT: {names}")
        
        if self.batch_generator is None:
            return None
            
        try:
            batch_data = next(self.batch_generator)
            
            if not self.dm_time_only:
                ft_batch, dt_batch, files = batch_data
                print(f"[DEBUG] FT shape: {ft_batch.shape}, DT shape: {dt_batch.shape}, files: {len(files)}")
                
                # Ensure data is contiguous and correct dtype - use ravel() like in inference code
                ft_contiguous = np.ascontiguousarray(ft_batch.astype(np.float32))
                dt_contiguous = np.ascontiguousarray(dt_batch.astype(np.float32))
                
                # Copy to GPU
                self.common.memcpy_host_to_device(self.ft_allocation, ft_contiguous)
                self.common.memcpy_host_to_device(self.dt_allocation, dt_contiguous)
                
                # IMPORTANT: Create pointer map and return in the EXACT order of 'names'
                ptr_map = {
                    'data_freq_time': int(self.ft_allocation),
                    'data_dm_time': int(self.dt_allocation),
                    'ft': int(self.ft_allocation),  # fallback names
                    'dt': int(self.dt_allocation)
                }
                
                # Return pointers in the order TensorRT expects
                result = [ptr_map.get(name) for name in names]
                
                # Ensure no None values
                if None in result:
                    print(f"[ERROR] Some input names not found: {names}")
                    return None
                    
                print(f"[DEBUG] Returning pointers in order: {names} -> {result}")
                return result
                
            else:
                dt_batch, files = batch_data
                print(f"[DEBUG] DT shape: {dt_batch.shape}, files: {len(files)}")
                
                dt_contiguous = np.ascontiguousarray(dt_batch.astype(np.float32))
                self.common.memcpy_host_to_device(self.dt_allocation, dt_contiguous)
                
                # Only DT input
                ptr_map = {
                    'data_dm_time': int(self.dt_allocation),
                    'dt': int(self.dt_allocation)
                }
                result = [ptr_map.get(name) for name in names]
                
                if None in result:
                    print(f"[ERROR] Some input names not found: {names}")
                    return None
                    
                print(f"[DEBUG] Returning pointers in order: {names} -> {result}")
                return result
                
            self.processed += len(files)
            print(f"[CALIBRATION] Processed {self.processed} / {self.total} files")
                    
        except StopIteration:
            print("[CALIBRATION] All calibration batches complete.")
            return None
        except Exception as e:
            print(f"[ERROR] In get_batch: {e}")
            import traceback
            traceback.print_exc()
            return None

    def read_calibration_cache(self):
        if self.cache_file is not None and os.path.exists(self.cache_file):
            print(f"[CALIBRATION] Loading cache from: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_file is None:
            return
        print(f"[CALIBRATION] Writing cache to: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ---------------------------------------------------------------------------
# Engine Builder
# ---------------------------------------------------------------------------

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine.
    Supported precisions: FP32, FP16, INT8, FP8, FP4
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

    def create_network(self, onnx_model_id, batch_size=1, dynamic_batch_size=None, local=False):
        """
        Parse ONNX and create the TensorRT network.

        :param onnx_model_id: ID of the ONNX model to download and parse.
        :param batch_size: Static batch size (used only if dynamic_batch_size is None).
        :param dynamic_batch_size: Comma-separated MIN,OPT,MAX or list of 3 ints.
                                   OPT is a tuning hint and is independent of calib batch size.
        :param local: Use local model file instead of downloading from model zoo.
        """
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        
        if not local:
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
                    # Static fallback within a dynamic profile
                    min_shape = opt_shape = max_shape = (batch_size, 256, 256, 1)

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
        calib_num_images=500,
        calib_batch_size=8,
    ):
        """
        Build and serialize the TensorRT engine.

        :param input_name: Name of the engine.
        :param calib_input: Directory containing H5 files for calibration (INT8 only).
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

        # --- Set precision flags and calibration (INT8 only) ---
        if precision == "int8":
            if self.builder.platform_has_fast_int8:
                print("[INFO] INT8 fast mode supported on this device.")
            else:
                print("[WARNING] INT8 is not natively supported on this device — may fall back to FP32.")
            self.config.set_flag(trt.BuilderFlag.INT8)
            
            # INT8 requires calibration
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

            self.config.int8_calibrator = EngineCalibrator(
                cache_file=calib_cache,
                h5_files=h5_files,
                calib_batch_size=calib_batch_size,
                dm_time_only=self.dm_time_only
            )
            
        elif precision == "fp16":
            if self.builder.platform_has_fast_fp16:
                print("[INFO] FP16 fast mode supported on this device.")
            else:
                print("[WARNING] FP16 is not natively supported on this device — may fall back to FP32.")
            self.config.set_flag(trt.BuilderFlag.FP16)
            # No calibration needed - weights are directly converted to FP16
            
        elif precision == "fp8":
            if self.builder.platform_has_fast_fp8:
                print("[INFO] FP8 fast mode supported on this device.")
            else:
                print("[WARNING] FP8 is not natively supported on this device — may fall back to FP32.")
            self.config.set_flag(trt.BuilderFlag.FP8)
            # No calibration needed - model must be exported with FP8 quantization
            
        elif precision == "fp4":
            if self.builder.platform_has_fast_fp4:
                print("[INFO] FP4 fast mode supported on this device.")
            else:
                print("[WARNING] FP4 is not natively supported on this device — may fall back to FP32.")
            self.config.set_flag(trt.BuilderFlag.FP4)
            # No calibration needed - model must be exported with FP4 quantization
            
        elif precision == "fp32":
            print("[INFO] Building FP32 engine (default precision).")
            # No flags needed - FP32 is the default

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
    builder = EngineBuilder(
        verbose=args.verbose, 
        workspace=args.workspace,
        dm_time_only=args.dm_time_only,
        precision=args.precision
    )
    builder.create_network(
        onnx_model_id=args.onnx,
        batch_size=args.batch_size,
        dynamic_batch_size=args.dynamic_batch_size,
        local=args.local
    )
    builder.create_engine(
        input_name=args.engine,
        calib_input=args.calib_input,
        calib_cache=args.calib_cache,
        calib_num_images=args.calib_num_images,
        calib_batch_size=args.calib_batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine for FETCH FRB detection."
    )
    parser.add_argument("-o", "--onnx", required=True,
                        help="INDEX of input ONNX model")
    parser.add_argument("-e", "--engine", required=True,
                        help="Output name for the TRT engine")
    parser.add_argument("-b", "--batch_size", default=1, type=int,
                        help="Static batch size (ignored if --dynamic_batch_size set), default: 1")
    parser.add_argument("-d", "--dynamic_batch_size", default=None,
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
    parser.add_argument("--precision", default="int8", choices=["int8", "fp16", "fp32", "fp8", "fp4"],
                        help="Precision mode to build in, default: int8")
    parser.add_argument("-D", "--dm_time_only", action="store_true",
                        help="Only use DM-Time data for inference (calibration only uses DT input)")
    parser.add_argument("--local", action="store_true",
                        help="Use local(custom) models instead of downloading from the model zoo")

    args = parser.parse_args()

    if args.calib_cache is None or not os.path.exists(args.calib_cache):
        if args.calib_input is None:
            parser.print_help()
            print("\n[ERROR] Provide --calib_input (H5 directory) or an existing --calib_cache.")
            sys.exit(1)

    main(args)