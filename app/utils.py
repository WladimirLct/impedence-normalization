import os
import pandas as pd
import glob
import regex as re
from io import StringIO
from datetime import datetime

# 3. Define Helper Functions
# These functions handle reading data files, extracting measurement times, and loading data for each well.

def read_data_file(filename, impedance_parameters=None):
    """
    Reads a data file and returns a pandas DataFrame.
    
    Parameters:
        filename (str): Path to the data file.
        impedance_parameters (list or None): List of impedance parameters to include. If None, include all.
        
    Returns:
        pd.DataFrame or None: DataFrame containing the data or None if reading fails.
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        # Identify the header line starting with 'Frequency'
        header_line_index = None
        for idx, line in enumerate(lines):
            if line.strip().startswith('Frequency'):
                header_line_index = idx
                break
        if header_line_index is None:
            print(f"No header found in file {filename}")
            return None
        # Extract header and data
        header_line = lines[header_line_index].strip()
        columns = re.split(r'\s+', header_line.replace('\t', ' '))
        data_str = ''.join(lines[header_line_index + 1:])
        data = pd.read_csv(
            StringIO(data_str),
            sep=r'\s+',
            names=columns,
            engine='python'
        )
        
        if impedance_parameters:
            # Filter columns to include only specified impedance parameters and Frequency
            columns_to_include = ['Frequency'] + [param for param in impedance_parameters if param in data.columns]
            data = data[columns_to_include]
        else:
            # Include all columns except 'Frequency' as parameters
            columns_to_include = [col for col in data.columns if col != 'Frequency']
        
        return data
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def extract_measurement_time_from_filename(filename):
    """
    Extracts measurement time from the filename.
    
    Expected filename format: 'YYYYMMDD_HH-MM-SS_...'
    
    Parameters:
        filename (str): Filename from which to extract the time.
        
    Returns:
        datetime or None: Extracted measurement time or None if parsing fails.
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) >= 2:
        datetime_str = parts[0] + '_' + parts[1]  # 'YYYYMMDD_HH-MM-SS'
        try:
            measurement_time = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')
            return measurement_time
        except ValueError:
            return None
    else:
        return None

def load_well_data(well_path, impedance_parameters=None):
    """
    Loads and processes impedance data for a specific well, including all frequencies.
    
    Parameters:
        well_path (str): Path to the well's data folder.
        impedance_parameters (list or None): List of impedance parameters to include. If None, include all.
        
    Returns:
        pd.DataFrame or None: DataFrame with Experiment, Well, Time, Frequency, and impedance parameters or None if no data found.
    """
    data_files = glob.glob(os.path.join(well_path, '*.txt'))
    if not data_files:
        print(f"No data files found in well path: {well_path}")
        return None

    records = []

    for data_file in data_files:
        data = read_data_file(data_file, impedance_parameters)
        if data is None:
            continue
        if 'Frequency' not in data.columns:
            print(f"'Frequency' column not found in file {data_file}")
            continue

        # Extract the measurement time
        measurement_time = extract_measurement_time_from_filename(data_file)
        if measurement_time is None:
            print(f"Could not extract measurement time from filename: {data_file}")
            continue

        base, _ = os.path.split(well_path)
        base, selected_experiment = os.path.split(base)
        
        # Melt the DataFrame to have one row per frequency and parameter
        melted_df = data.melt(id_vars=['Frequency'], var_name='Parameter', value_name='Value')
        melted_df['Experiment'] = selected_experiment
        melted_df['Well'] = os.path.basename(well_path)
        melted_df['Date'] = measurement_time

        # Reorder columns
        melted_df = melted_df[['Experiment', 'Well', 'Date', 'Frequency', 'Parameter', 'Value']]

        records.append(melted_df)

    if records:
        combined_well_df = pd.concat(records, ignore_index=True)
        return combined_well_df
    else:
        print(f"No valid data extracted for well at {well_path}")
        return None

def check_data(combined_df):
    # 5. Handle Missing Values and Save the Combined DataFrame to a CSV File
    # This step ensures that the DataFrame is complete and saves it to a CSV file for later use.
    
    # Check for missing values
    missing_values = combined_df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values)
    
    # Handle missing values if any
    if missing_values.any():
        # Depending on the nature of missing data, choose an appropriate method
        # For this example, we'll forward-fill and backward-fill
        combined_df.sort_values(['Experiment', 'Well', 'Date', 'Frequency', 'Parameter'], inplace=True)
        combined_df.fillna(method='ffill', inplace=True)
        combined_df.fillna(method='bfill', inplace=True)
    
        # Verify that all missing values are handled
        if combined_df.isnull().sum().any():
            print("Warning: There are still missing values after filling.")
        else:
            print("All missing values have been handled.")
    
    # Optionally, pivot the DataFrame to have Parameters as separate columns
    # This depends on downstream requirements
    pivot_df = combined_df.pivot_table(
        index=['Experiment', 'Well', 'Date', 'Frequency'],
        columns='Parameter',
        values='Value'
    ).reset_index()
    
    # Flatten the MultiIndex columns if necessary
    pivot_df.columns.name = None
    pivot_df.columns = [str(col) for col in pivot_df.columns]
    
    return pivot_df
