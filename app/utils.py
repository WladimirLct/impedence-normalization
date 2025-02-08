import pandas as pd

from io import StringIO
from datetime import datetime   

from concurrent.futures import ProcessPoolExecutor

import re
import os

from normalizations import normalize_columns

def load_wells(folder_path, save_path, impedance_parameters=None):
    print(f"Loading data from folder: {folder_path}")
    start = datetime.now()
    well_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    wells_data = []
    for well_path in well_paths:
        wells_data.append(load_well_data(well_path, impedance_parameters))
    end = datetime.now()
    print(f"Loading time: {end-start}")

    start = datetime.now()
    if wells_data:
        combined_data = pd.concat(wells_data, ignore_index=True)
        combined_data = normalize_columns(combined_data, save_path)
    end = datetime.now()
    print(f"Normalization time: {end-start}")
    return combined_data

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

def process_file(args):
    data_file, selected_experiment, well_name, impedance_parameters = args

    data = read_data_file(data_file, impedance_parameters)
    if data is None or 'Frequency' not in data.columns:
        return None

    measurement_time = extract_measurement_time_from_filename(data_file)
    if measurement_time is None:
        return None

    data['Experiment'] = selected_experiment
    data['Well'] = well_name
    data['Date'] = measurement_time

    # Reshape the DataFrame
    melted_df = data.melt(
        id_vars=['Experiment', 'Well', 'Date', 'Frequency'],
        var_name='Parameter',
        value_name='Value'
    )

    # Convert Value and Frequency to numerics
    melted_df['Value'] = pd.to_numeric(melted_df['Value'], errors='coerce')
    melted_df['Frequency'] = pd.to_numeric(melted_df['Frequency'], errors='coerce')

    return melted_df

def load_well_data(well_path, impedance_parameters=None):
    data_files = [os.path.join(well_path, f) for f in os.listdir(well_path) if f.endswith('.txt')]
    if not data_files:
        print(f"No data files found in well path: {well_path}")
        return None

    base, _ = os.path.split(well_path)
    base, selected_experiment = os.path.split(base)
    well_name = os.path.basename(well_path)

    args_list = [
        (data_file, selected_experiment, well_name, impedance_parameters)
        for data_file in data_files
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, args_list))

    records = [df for df in results if df is not None]

    if records:
        combined_well_df = pd.concat(records, ignore_index=True)
        return combined_well_df
    else:
        print(f"No valid data extracted for well at {well_path}")
        return None