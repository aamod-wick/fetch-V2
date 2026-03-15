"""
TensorRT INT8 Engine Builder for FETCH Fast Radio Burst Detection.
Calibration uses H5 files containing freq-time and DM-time data.
"""

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
from model_handler import download_model # For potential future use of model registry or ONNX paths


# ---------------------------------------------------------------------------
# H5 Data Preprocessing
# ---------------------------------------------------------------------------

def preprocess_ft_data(data):
    data = np.nan_to_num(data.astype(np.float32))
    data = s.detrend(data)
    data = data - np.median(data)
    std = np.std(data)
    if std > 0:
        data = data / std
    return np.nan_to_num(data)


def preprocess_dt_data(data):
    data = np.nan_to_num(data.astype(np.float32))
    data = data - np.median(data)
    std = np.std(data)
    if std > 0:
        data = data / std
    return np.nan_to_num(data)


def load_and_preprocess_h5(h5_path, ft_dim=(256, 256), dt_dim=(256, 256)):
    """
    Load and preprocess a single H5 file.
    Returns ft_data and dt_data each with shape (H, W, 1).
    """
    with h5py.File(h5_path, "r") as f:
        ft_raw = np.array(f["data_freq_time"], dtype=np.float32).T
        dt_raw = np.array(f["data_dm_time"], dtype=np.float32)

    ft = preprocess_ft_data(ft_raw)
    dt = preprocess_dt_data(dt_raw)

    ft = np.reshape(ft, (*ft_dim, 1))
    dt = np.reshape(dt, (*dt_dim, 1))
    return ft, dt


def h5_batch_generator(h5_files, batch_size=8):
    """
    Yields (ft_batch, dt_batch, file_paths) for each batch of H5 files.
    ft_batch and dt_batch have shape (N, 256, 256, 1).
    Skips files that fail to load.
    """
    for i in range(0, len(h5_files), batch_size):
        batch_files = h5_files[i: i + batch_size]
        ft_list, dt_list, valid = [], [], []

        for path in batch_files:
            try:
                ft, dt = load_and_preprocess_h5(path)
                ft_list.append(ft)
                dt_list.append(dt)
                valid.append(path)
            except Exception as e:
                print(f"[WARNING] Skipping {path}: {e}")

        if valid:
            yield np.array(ft_list), np.array(dt_list), valid


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

    def __init__(self, cache_file, h5_files, calib_batch_size=8):
        super().__init__()
        self.cache_file = cache_file
        self.calib_batch_size = calib_batch_size
        self.total = len(h5_files)
        self.processed = 0
        self.common = Common()

        # GPU allocations for both inputs — fixed shape (B, 256, 256, 1) float32
        size = int(np.dtype(np.float32).itemsize * calib_batch_size * 256 * 256 * 1)
        self.ft_allocation = self.common.cuda_call(cudart.cudaMalloc(size))
        self.dt_allocation = self.common.cuda_call(cudart.cudaMalloc(size))

        # Wire up the generator only if there are files to process
        if h5_files:
            self.batch_generator = h5_batch_generator(h5_files, batch_size=calib_batch_size)
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
        # Uncomment once to verify input order during first run:
        # print(f"[DEBUG] Calibration input names from TRT: {names}")

        try:
            ft_batch, dt_batch, files = next(self.batch_generator)
            self.processed += len(files)
            print(f"[CALIBRATION] Processed {self.processed} / {self.total} files")

            self.common.memcpy_host_to_device(
                self.ft_allocation, np.ascontiguousarray(ft_batch)
            )
            self.common.memcpy_host_to_device(
                self.dt_allocation, np.ascontiguousarray(dt_batch)
            )

            # Order: ft first, dt second — must match ONNX export input order
            return [int(self.ft_allocation), int(self.dt_allocation)]

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
    Parses an ONNX graph and builds a TensorRT INT8 engine.
    """

    def __init__(self, verbose=False, workspace=8):
        self.trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30)
        )

        self.network = None
        self.parser = None

    def create_network(self, onnx_path, batch_size=1, dynamic_batch_size=None):
        """
        Parse ONNX and create the TensorRT network.

        :param onnx_path: Path to the ONNX model.
        :param batch_size: Static batch size (used only if dynamic_batch_size is None).
        :param dynamic_batch_size: Comma-separated MIN,OPT,MAX or list of 3 ints.
                                   OPT is a tuning hint and is independent of calib batch size.
        """
        self.network = self.builder.create_network(0)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
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
        engine_path,
        calib_input=None,
        calib_cache=None,
        calib_num_images=500,
        calib_batch_size=8,
    ):
        """
        Build and serialize the INT8 TensorRT engine.

        :param engine_path: Output path for the serialized engine.
        :param calib_input: Directory containing H5 files for calibration.
        :param calib_cache: Path to read/write the INT8 calibration cache.
        :param calib_num_images: Max number of H5 files to use for calibration.
        :param calib_batch_size: Samples per calibration forward pass.
        """
        engine_path = os.path.realpath(engine_path)
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)

        if not self.builder.platform_has_fast_int8:
            print("[WARNING] INT8 is not natively supported on this device — may fall back to FP32.")

        self.config.set_flag(trt.BuilderFlag.INT8)

        # --- Set up calibrator ---
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
        )

        # --- Build ---
        print(f"[INFO] Building INT8 engine -> {engine_path}")
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
    builder = EngineBuilder(verbose=args.verbose, workspace=args.workspace)
    builder.create_network(
        onnx_path=args.onnx,
        batch_size=args.batch_size,
        dynamic_batch_size=args.dynamic_batch_size,
    )
    builder.create_engine(
        engine_path=args.engine,
        calib_input=args.calib_input,
        calib_cache=args.calib_cache,
        calib_num_images=args.calib_num_images,
        calib_batch_size=args.calib_batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a TensorRT INT8 engine for FETCH FRB detection."
    )
    parser.add_argument("-o", "--onnx", required=True,
                        help="Path to input ONNX model")
    parser.add_argument("-e", "--engine", required=True,
                        help="Output path for the TRT engine")
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

    args = parser.parse_args()

    if args.calib_cache is None or not os.path.exists(args.calib_cache):
        if args.calib_input is None:
            parser.print_help()
            print("\n[ERROR] Provide --calib_input (H5 directory) or an existing --calib_cache.")
            sys.exit(1)

    main(args)
#below is the hardcoded test / demo entry ; swap with the entry point above if you want to test the INT8 engine
'''
if __name__ == "__main__":
    # Set to True to run the hardcoded Colab-friendly demo instead of CLI parsing.
    RUN_DEMO = True

    if RUN_DEMO:
        class DemoArgs:
            # Hardcoded paths — adjust to your Colab paths
            onnx = "/content/model.onnx"
            engine = "/content/model_int8.engine"
            # Static batch (ignored if dynamic_batch_size is set)
            batch_size = 1
            # Use None for static, or e.g. "1,4,8" for dynamic
            dynamic_batch_size = None
            verbose = True
            # Workspace in GB
            workspace = 4
            # Calibration data (directory of .h5 files) and cache path
            calib_input = "/content/calib_h5"
            calib_cache = "/content/calib.cache"
            # Smaller numbers for quick Colab runs
            calib_num_images = 16
            calib_batch_size = 4

        main(DemoArgs())
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Build a TensorRT INT8 engine for FETCH FRB detection."
    )
    '''