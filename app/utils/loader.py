# File: utils/loader.py

import os
import re
import pandas as pd

from threading import Thread, Lock
from io import StringIO
from datetime import datetime

from config.cache import cache

def format_path(base_path):
    return base_path.replace("\\", "/")

def update_well_list(df):
    wells = df["well"].unique().tolist()
    cache["wells"] = wells
    return

def list_folders(base_path) -> pd.DataFrame:
    clean_path = format_path(base_path)

    files = [f for f in os.listdir(base_path)]
    txt_files = [os.listdir(clean_path + "/" + file) for file in files]

    files_dict = {
        "base_dir": [],
        "well": [],
        "measure": [],
    }

    for i in range(len(files)):
        for f in txt_files[i]:
            files_dict["base_dir"].append(clean_path)
            files_dict["well"].append(files[i])
            files_dict["measure"].append(f)

    df = pd.DataFrame(files_dict)
    update_well_list(df)

    return df

# Global variables for thread-safe data loading
loading_data = []
loading_data_lock = Lock()

def extract_measurement_time_from_filename(filename):
    """
    Extracts measurement time from the filename.
    Expected filename format: 'YYYYMMDD_HH-MM-SS_...'
    """
    base = os.path.basename(filename)
    match = re.match(r'(\d{8}_\d{2}-\d{2}-\d{2})', base)
    if match:
        datetime_str = match.group(1)
        try:
            measurement_time = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')
            return measurement_time
        except ValueError:
            print(f"Invalid date format in file {filename}")
            return None
    else:
        print(f"Filename does not contain date information: {filename}")
        return None

def read_data_file(filename, impedance_parameters=None):
    """
    Reads a data file and returns a pandas DataFrame.
    """
    try:
        with open(filename, 'r') as f:
            # Read lines until you find the header
            for line in f:
                if line.strip().startswith('Frequency'):
                    header_line = line.strip()
                    break
            else:
                print(f"No header found in file {filename}")
                return None

            # Read the rest of the file from the current position
            data_str = f.read()

        # Use optimized parsing
        columns = re.split(r'\s+', header_line)
        data = pd.read_csv(
            StringIO(data_str),
            sep=r'\s+',
            names=columns,
            engine='c',  # Use C engine for faster parsing
            na_filter=False
        )

        if impedance_parameters:
            columns_to_include = ['Frequency'] + [param for param in impedance_parameters if param in data.columns]
            data = data[columns_to_include]
        else:
            # Ensure 'Frequency' is included
            data = data[['Frequency'] + [col for col in data.columns if col != 'Frequency']]

        return data
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def loop_files(low_bound, high_bound):
    """
    Processes a range of files and updates the global loading_data list.
    """
    file_df = cache["paths"].iloc[low_bound:high_bound]

    for idx, data in file_df.iterrows():
        path = os.path.join(data["base_dir"], data["well"], data["measure"])
        try:
            # Read the data file
            file_data = read_data_file(path)

            if file_data is not None:
                # Extract additional metadata
                measurement_time = extract_measurement_time_from_filename(path)
                if measurement_time is not None:
                    file_data['Date'] = measurement_time

                # Add experiment and well information
                file_data['Experiment'] = os.path.basename(data["base_dir"])
                file_data['Well'] = data["well"]

                # Convert numeric columns
                file_data['Frequency'] = pd.to_numeric(file_data['Frequency'], errors='coerce')
                for col in file_data.columns:
                    if col not in ['Experiment', 'Well', 'Date']:
                        file_data[col] = pd.to_numeric(file_data[col], errors='coerce')

                # Thread-safe update to the global loading_data list
                with loading_data_lock:
                    loading_data.append(file_data)

        except Exception as e:
            print(f"Error processing file {path}: {e}")

def load_files(num_workers=4):
    """
    Loads files using multiple threads and updates the global loading_data list.
    """
    with loading_data_lock:
        loading_data.clear()

    file_count = len(cache["paths"])
    file_count_per_worker = (file_count // num_workers) + 1

    threads = []

    for i in range(num_workers):
        low_bound = i * file_count_per_worker
        high_bound = (i + 1) * file_count_per_worker if (i + 1) * file_count_per_worker < file_count else file_count

        thread = Thread(target=loop_files, args=(low_bound, high_bound))
        thread.start()
        threads.append(thread)