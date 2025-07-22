# File: utils/loader.py

import pandas as pd
import numpy as np

import time
from datetime import datetime

from threading import Thread, Lock
from multiprocessing import Pool

from config.cache import cache

# Global variables for thread-safe normalization progress tracking
normalization_progress = {
    'started': False,
    'finished': False,
    'current_step': 0,
    'total_steps': 8,
    'step_description': '',
    'error': None
}
normalization_progress_lock = Lock()

def reset_normalization_progress():
    """Reset the normalization progress to initial state"""
    with normalization_progress_lock:
        normalization_progress.update({
            'started': False,
            'finished': False,
            'current_step': 0,
            'total_steps': 8,
            'step_description': '',
            'error': None
        })

def update_normalization_progress(step, description, finished=False, error=None):
    """Thread-safe update of normalization progress"""
    with normalization_progress_lock:
        normalization_progress['current_step'] = step
        normalization_progress['step_description'] = description
        normalization_progress['started'] = True
        normalization_progress['finished'] = finished
        if error:
            normalization_progress['error'] = error

def get_normalization_progress():
    """Thread-safe getter for normalization progress"""
    with normalization_progress_lock:
        return normalization_progress.copy()

def start_normalization_thread(loading_data, save_path):
    """Start normalization in a separate thread"""
    def normalization_worker():
        try:
            reset_normalization_progress()
            normalize_columns(loading_data, save_path)
        except Exception as e:
            update_normalization_progress(0, f"Error: {str(e)}", finished=True, error=str(e))
    
    thread = Thread(target=normalization_worker)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    return thread

def process_frequency(args):
    frq, well, df_frq_well, fmt = args

    # Convert Date back to datetime for processing
    df_frq_well["Date"] = pd.to_datetime(df_frq_well["Date"], format=fmt)

    # Sort by date if not already sorted (avoid redundant sorting)
    if not df_frq_well["Date"].is_monotonic_increasing:
        df_frq_well = df_frq_well.sort_values("Date").reset_index(drop=True)

    # Check if AbsZ column exists
    if 'AbsZ' not in df_frq_well.columns:
        #print(f"[WARNING] AbsZ column not found for frequency {frq}, well {well}. Available columns: {df_frq_well.columns.tolist()}")
        return df_frq_well

    # Get AbsZ values as numpy array for faster operations
    absz_values = df_frq_well["AbsZ"].values
    
    # Calculate min/max once
    absz_max = absz_values.max()
    absz_min = absz_values.min()
    absz_t0 = absz_values[0]
    
    # Avoid division by zero
    absz_range = absz_max - absz_min
    if absz_range == 0:
        absz_range = 1

    # Vectorized normalized calculations
    df_frq_well["AbsZ_t0"] = absz_values - absz_t0
    df_frq_well["AbsZ_max"] = absz_values / absz_max if absz_max != 0 else 0
    df_frq_well["AbsZ_min_max"] = (absz_values - absz_min) / absz_range

    # Convert Date back to string format for consistency
    df_frq_well["Date"] = df_frq_well["Date"].dt.strftime(fmt)

    return df_frq_well

def fix_data(df):
    #print(f"[DEBUG] Starting fix_data with {len(df)} rows")
    start_time = time.time()
    
    # Check for missing values more efficiently
    if df.isnull().any().any():
        #print(f"[DEBUG] Found missing values, applying forward/backward fill")
        # Sort only if we have missing values
        df = df.sort_values(['Experiment', 'Well', 'Date', 'Frequency', 'Parameter'])
        
        # Use the new pandas 2.0+ approach for filling missing values
        df = df.ffill().bfill()
    # else:
        #print(f"[DEBUG] No missing values found")
    
    #print(f"[DEBUG] Starting pivot operation")
    pivot_start = time.time()
    
    # Use pivot_table with aggfunc='first' to handle potential duplicates
    df_pivoted = df.pivot_table(
        index=['Experiment', 'Well', 'Date', 'Frequency'],
        columns='Parameter',
        values='Value',
        aggfunc='first'  # Handle potential duplicates
    ).reset_index()
    
    # Clean column names (remove the name from columns)
    df_pivoted.columns.name = None
    
    pivot_time = time.time() - pivot_start
    total_time = time.time() - start_time
    #print(f"[DEBUG] Pivot completed in {pivot_time:.2f}s, total fix_data time: {total_time:.2f}s")
    
    return df_pivoted

def normalize_columns(loading_data, save_path):
    #print(f"\n{'='*60}")
    #print(f"[NORMALIZE] Starting normalization process")
    #print(f"[NORMALIZE] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    overall_start = time.time()
    
    if not loading_data:
        #print("[ERROR] No data to normalize")
        update_normalization_progress(0, "no data to normalize", finished=True, error="No data provided")
        return pd.DataFrame()
    
    #print(f"[NORMALIZE] Input: {len(loading_data)} DataFrames to process")
    
    # Step 1: Concatenate data
    #print(f"[STEP 1/6] Concatenating DataFrames...")
    update_normalization_progress(1, "concaténation des données")
    step_start = time.time()
    df_combined = pd.concat(loading_data, ignore_index=True, copy=False)
    step_time = time.time() - step_start
    #print(f"[STEP 1/6] ✓ Combined into {len(df_combined)} rows in {step_time:.2f}s")
    #print(f"[DEBUG] Combined DataFrame columns: {df_combined.columns.tolist()}")
    
    # Step 2: Check melting necessity
    #print(f"[STEP 2/6] Checking data structure...")
    update_normalization_progress(2, "vérification de la structure des données")
    step_start = time.time()
    non_id_cols = [col for col in df_combined.columns 
                   if col not in ['Experiment', 'Well', 'Date', 'Frequency']]
    
    if len(non_id_cols) <= 1:
        #print(f"[STEP 2/6] ✓ Data already in correct format ({len(non_id_cols)} parameter columns)")
        df = df_combined
    else:
        #print(f"[STEP 2/6] Melting {len(non_id_cols)} parameter columns: {non_id_cols}")
        # Melt the dataframe
        df = df_combined.melt(
            id_vars=['Experiment', 'Well', 'Date', 'Frequency'],
            value_vars=non_id_cols,
            var_name='Parameter',
            value_name='Value'
        )
        
        # Convert to numeric more efficiently
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        #print(f"[STEP 2/6] ✓ Melted to {len(df)} rows")
    
    step_time = time.time() - step_start
    #print(f"[STEP 2/6] Completed in {step_time:.2f}s")
    
    # Step 3: Ensure numeric types
    #print(f"[STEP 3/6] Converting data types...")
    update_normalization_progress(3, "conversion des types de données")
    time.sleep(0.5)
    step_start = time.time()
    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
    step_time = time.time() - step_start
    #print(f"[STEP 3/6] ✓ Completed in {step_time:.2f}s")
    
    # Step 4: Fix data
    #print(f"[STEP 4/6] Fixing data (pivot, fill missing values)...")
    update_normalization_progress(4, "correction des données (pivot, valeurs manquantes) ")
    step_start = time.time()
    df = fix_data(df)
    step_time = time.time() - step_start
    #print(f"[STEP 4/6] ✓ Fixed data: {len(df)} rows in {step_time:.2f}s")
    #print(f"[DEBUG] Fixed DataFrame columns: {df.columns.tolist()}")
    
    # Step 5: Process groups
    #print(f"[STEP 5/6] Processing frequency-well groups...")
    update_normalization_progress(5, "filtrage des groupes fréquence-puits")
    step_start = time.time()
    
    # Ensure Date is datetime for grouping operations
    if df["Date"].dtype == 'object':
        #print(f"[DEBUG] Converting Date column from object to datetime")
        df["Date"] = pd.to_datetime(df["Date"])
    
    # Create groups more efficiently
    grouped = df.groupby(['Frequency', 'Well'], sort=False)
    total_groups = len(grouped)
    #print(f"[STEP 5/6] Found {total_groups} frequency-well combinations to process")
    
    # Pre-allocate list for better memory management
    freq_well_dfs = []
    fmt = '%Y%m%d_%H-%M-%S'
    
    for i, ((frq, well), group) in enumerate(grouped):
        if i % max(1, total_groups // 10) == 0:  # Progress every 10%
            progress = (i / total_groups) * 100
            #print(f"[STEP 5/6] Progress: {progress:.1f}% ({i}/{total_groups} groups)")
        
        # Convert dates to string format only when needed
        group_copy = group.copy()
        group_copy["Date"] = group_copy["Date"].dt.strftime(fmt)
        group_copy['original_index'] = group.index
        freq_well_dfs.append((frq, well, group_copy, fmt))
    
    step_time = time.time() - step_start
    #print(f"[STEP 5/6] ✓ Prepared {len(freq_well_dfs)} groups in {step_time:.2f}s")
    
    # Step 6: Multiprocessing
    #print(f"[STEP 6/6] Processing groups with multiprocessing...")
    update_normalization_progress(6, "calcul des normalisations")
    step_start = time.time()
    
    with Pool() as pool:
        #print(f"[STEP 6/6] Using {pool._processes} processes")
        dfs = pool.map(process_frequency, freq_well_dfs)
    
    processing_time = time.time() - step_start
    #print(f"[STEP 6/6] ✓ Multiprocessing completed in {processing_time:.2f}s")
    
    # Final assembly
    #print(f"[FINAL] Assembling final DataFrame...")
    update_normalization_progress(7, "finalisation")
    # final_start = time.time()
    
    if dfs:
        df_processed = pd.concat(dfs, ignore_index=True)
        df_processed = df_processed.sort_values('original_index').drop(columns=['original_index'])
        
        # Convert back to datetime
        df_processed["Date"] = pd.to_datetime(df_processed["Date"], format=fmt)
        df_processed.columns = df_processed.columns.str.lower()

        # Save to CSV
        print(f"[FINAL] Saving to {save_path}...")
        # save_start = time.time()

        cache["data"] = df_processed
        df_processed.to_csv(save_path, index=False)
        
        # save_time = time.time() - save_start
        
        # final_time = time.time() - final_start
        # total_time = time.time() - overall_start
        
        #print(f"[SUCCESS] Normalization completed!")
        #print(f"[SUCCESS] Total time: {total_time:.2f}s")
        #print(f"[SUCCESS] Final DataFrame: {len(df_processed)} rows, {len(df_processed.columns)} columns")
        
        update_normalization_progress(6, "normalization completed successfully", finished=True)
        return df_processed
    else:
        #print(f"[ERROR] No data processed successfully")
        update_normalization_progress(0, "no data processed successfully", finished=True, error="Processing failed")
        return pd.DataFrame()
    
def normalize_t0(data):
    """
    Normalize data so that all wells start at value 1 at t0.
    Divides each well's values by its first (t0) value.
    Returns data with a new 'absz_t0' column.
    """
    data = data.copy()
    
    # Find the baseline (t0) value for each well
    baseline_values = (
        data
        .groupby('well')['elapsed_hours']
        .idxmin()  # Get index of minimum elapsed_hours for each well
        .to_dict()
    )
    
    # Create the normalized column
    data['absz_t0'] = data['absz']  # Start with original values
    
    # Normalize each well by its t0 value
    for well in data['well'].unique():
        well_data = data[data['well'] == well]
        if len(well_data) > 0:
            # Get the baseline value (first timepoint for this well)
            baseline_idx = baseline_values.get(well)
            if baseline_idx is not None:
                baseline_value = data.loc[baseline_idx, 'absz']
                if baseline_value != 0:  # Avoid division by zero
                    data.loc[data['well'] == well, 'absz_t0'] = (
                        data.loc[data['well'] == well, 'absz'] / baseline_value
                    )
    
    return data