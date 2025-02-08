import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

def fix_data(df):
    missing_values = df.isnull().sum()

    # Handle missing values if any
    if missing_values.any():
        # For this example, we'll forward-fill and backward-fill
        df.sort_values(['Experiment', 'Well', 'Date', 'Frequency', 'Parameter'], inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
    else:
        print("No missing values found")

    # Pivot the DataFrame
    df = df.pivot_table(
        index=['Experiment', 'Well', 'Date', 'Frequency'],
        columns='Parameter',
        values='Value'
    ).reset_index()

    return df

def process_frequency(args):
    frq, well, df_frq_well, fmt = args  # Unpack the arguments

    # Ensure 'Date' is datetime
    df_frq_well["Date"] = pd.to_datetime(df_frq_well["Date"], format=fmt)

    # Sort by 'Date' to have the correct order
    df_frq_well = df_frq_well.sort_values("Date").reset_index(drop=True)

    # Compute absz_max, absz_min, t0, absz_t0
    absz_max = df_frq_well["AbsZ"].max()
    absz_min = df_frq_well["AbsZ"].min()
    t0 = df_frq_well["Date"].iloc[0]
    absz_t0 = df_frq_well["AbsZ"].iloc[0]

    # Compute time deltas from t0
    dt = df_frq_well["Date"] - t0

    # Compute 'index' as a sequential integer
    df_frq_well["index"] = df_frq_well.index

    # Compute total seconds
    total_seconds = dt.dt.total_seconds()

    # Compute 'hours', 'minutes', 'seconds' using vectorized operations
    df_frq_well["hours"] = total_seconds / 3600
    remaining_seconds = total_seconds % 3600
    df_frq_well["minutes"] = remaining_seconds / 60
    df_frq_well["seconds"] = remaining_seconds % 60

    # Compute AbsZ_t0, AbsZ_max, AbsZ_min_max using vectorized operations
    absz = df_frq_well["AbsZ"].values
    df_frq_well["AbsZ_t0"] = absz - absz_t0
    df_frq_well["AbsZ_max"] = absz / absz_max
    df_frq_well["AbsZ_min_max"] = (absz - absz_min) / (absz_max - absz_min)

    return df_frq_well

def normalize_columns(df, save_path):
    df = fix_data(df)
    fmt = '%Y%m%d_%H-%M-%S'
    df["Date"] = df["Date"].dt.strftime(fmt)

    # Reset index to ensure unique indices
    df = df.reset_index(drop=True)
    # Save the original indices
    df['original_index'] = df.index

    # Group df by 'Frequency' and 'Well'
    freq_well_groups = df.groupby(['Frequency', 'Well'])

    # Create a list of (frq, well, df_frq_well, fmt) tuples for multiprocessing
    freq_well_dfs = [(frq, well, group.copy(), fmt) for (frq, well), group in freq_well_groups]

    # Use multiprocessing Pool to process frequency-well groups in parallel
    with Pool() as pool:
        dfs = pool.map(process_frequency, freq_well_dfs)

    # Concatenate all processed DataFrames
    df_processed = pd.concat(dfs)
    # Sort df_processed by 'original_index' to maintain the original order
    df_processed = df_processed.sort_values('original_index')
    # Drop the temporary 'original_index' column
    df_processed = df_processed.drop(columns=['original_index'])

    # Set the date column back to datetime format
    df_processed["Date"] = pd.to_datetime(df_processed["Date"], format=fmt)

    # Save the processed DataFrame to a CSV file
    df_processed.to_csv(save_path, index=False)

    return df_processed
