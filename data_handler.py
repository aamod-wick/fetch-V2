import pandas as pd
import numpy as np
import ctypes
import scipy.signal as s
import os
import argparse
import glob
import hashlib
import logging
import string
from pathlib import Path
import h5py
import requests
import logging

# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


def preprocess_ft_data(data):
    # Replace NaNs and convert to float32
    data = np.nan_to_num(data.astype(np.float32))

    # Apply detrending (remove linear trend)
    data = s.detrend(data)

    # Normalize: subtract median, divide by standard deviation
    data = data - np.median(data)
    data = data / np.std(data)

    # Handle any remaining NaNs (in case std was 0)
    data = np.nan_to_num(data)

    return data


def preprocess_dt_data(data):
    """
    Apply DT (DM-Time) preprocessing pipeline.
    Extracted from DataGenerator.__data_generation.
    """
    # Replace NaNs and convert to float32
    data = np.nan_to_num(data.astype(np.float32))

    # Normalize: subtract median, divide by standard deviation
    data = data - np.median(data)
    data = data / np.std(data)

    # Handle any remaining NaNs (in case std was 0)
    data = np.nan_to_num(data)

    return data


def load_and_preprocess_h5_data(h5_file_path, ft_dim=(256, 256), dt_dim=(256, 256)):
    """
    Load and preprocess data from H5 file.

    :param h5_file_path: Path to H5 file
    :param ft_dim: Expected FT data dimensions
    :param dt_dim: Expected DT data dimensions
    :return: Tuple of (ft_data, dt_data) with shape (H, W, 1)
    """
    try:
        with h5py.File(h5_file_path, "r") as f:
            # Load raw data
            data_ft_raw = np.array(f["data_freq_time"], dtype=np.float32).T
            data_dt_raw = np.array(f["data_dm_time"], dtype=np.float32)

            # Apply preprocessing
            data_ft = preprocess_ft_data(data_ft_raw)
            data_dt = preprocess_dt_data(data_dt_raw)

            # Reshape to expected dimensions with channel dimension
            ft_data = np.reshape(data_ft, (*ft_dim, 1))
            dt_data = np.reshape(data_dt, (*dt_dim, 1))

            return ft_data, dt_data

    except Exception as e:
        logger.error(f"Failed to load/preprocess {h5_file_path}: {str(e)}")
        raise
def sort_h5_files_by_dm(h5_files,result_dir="sorted_files"):
    """
    Sort H5 files based on DM value extracted from file name.Stores files in result_dir in folder result_dir/{DM_value}/. Creates directories if they don't exist. Returns sorted list of file paths.
    Enable recursive call 
    :param h5_files: List of H5 file paths / directory
    :return: nothing 
    """
    
    try:
        # If h5_files is a directory, get all H5 files in it
        if isinstance(h5_files, str) and os.path.isdir(h5_files):
            h5_files = glob.glob(os.path.join(h5_files, "**", "*.h5"), recursive=True)

        # Create result directory if it doesn't exist (absolute path)
        os.makedirs(result_dir, exist_ok=True)
        sorted_files = []
        for h5_file in h5_files:
            dm_value = find_dm_of_file(h5_file)
            dm_dir = os.path.join(result_dir, f"DM_{dm_value:.2f}")
            os.makedirs(dm_dir, exist_ok=True)
            dest_path = os.path.join(dm_dir, os.path.basename(h5_file))
            os.rename(h5_file, dest_path)
            sorted_files.append(dest_path)

        return sorted_files
    except Exception as e:
        logger.error(f"Failed to sort H5 files: {str(e)}")
        raise
def find_dm_of_file(file_path):
    """
    Extract DM value from file name.

    :param file_path: Path to H5 file
    :return: Extracted DM value as float
    """
    try:
        filename = os.path.basename(file_path)
        # Assuming filename format is like "data_dm_123.45.h5"
        dm_str = filename.split("_dm_")[1].split(".h5")[0]
        return float(dm_str)
    except Exception as e:
        logger.error(f"Failed to extract DM from {file_path}: {str(e)}")
        raise
def process_batch(h5_files, batch_size=8, ft_dim=(256, 256), dt_dim=(256, 256)):
    """
    Process a batch of H5 files and return preprocessed data.

    :param h5_files: List of H5 file paths
    :param batch_size: Batch size for processing
    :param ft_dim: FT data dimensions
    :param dt_dim: DT data dimensions
    :return: Generator yielding (ft_batch, dt_batch, file_paths_batch)
    """
    for i in range(0, len(h5_files), batch_size):
        batch_files = h5_files[i : i + batch_size]

        ft_batch = []
        dt_batch = []
        valid_files = []

        for h5_file in batch_files:
            try:
                ft_data, dt_data = load_and_preprocess_h5_data(h5_file, ft_dim, dt_dim)
                ft_batch.append(ft_data)
                dt_batch.append(dt_data)
                valid_files.append(h5_file)
            except Exception as e:
                logger.warning(f"Skipping {h5_file}: {str(e)}")
                continue

        if valid_files:
            # Convert to numpy arrays with batch dimension
            ft_batch = np.array(ft_batch)
            dt_batch = np.array(dt_batch)

            yield ft_batch, dt_batch, valid_files
  ##---------------------------unit testing code---------------------------------##
#data_files = ["test.h5"]
#ft_batch, dt_batch, valid_files = process_batch(data_files)
sort_h5_files_by_dm("/home/aamod/spotlight-data","/home/aamod/sorted_spotlight_data_files")