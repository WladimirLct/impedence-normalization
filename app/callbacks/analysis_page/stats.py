# File: callbacks/analysis_page/stats.py

import pandas as pd
import numpy as np
from dash import callback, Output, Input, State, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from scipy.stats import ttest_ind, mannwhitneyu

import plotly.colors as pcolors
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

from config.cache import cache
from utils.tools import format_group_name
from utils.stats import bin_time_data, plot_stats, save_stats_info_to_file

def get_interpolated_groups_from_cache(well_groups_data):
    """
    Convert cached interpolated groups to the format expected by stats analysis.
    Returns a dictionary with group_id as key and interpolated DataFrame as value.
    """
    cached_interpolated = cache.get("interpolated_groups", [])
    
    if not cached_interpolated:
        return {}
    
    interpolated_groups = {}
    
    for cached_group in cached_interpolated:
        group_id = cached_group.get('id')  # Use cached ID directly
        interpolated_wells = cached_group.get('interpolated_wells')
        
        if group_id and interpolated_wells is not None:
            interpolated_groups[group_id] = interpolated_wells
    
    return interpolated_groups

def process_comparison_group(args):
    """
    Process a single comparison group analysis.
    This function runs in a separate thread.
    """
    (comp_id, groups, ref_data, ref_name, interpolated_groups, 
     p_threshold, y_param, color, idx, test_type) = args

    thread_id = f"Thread-{idx}"
    print(f"🧵 {thread_id}: Starting analysis for comparison group {comp_id}")
    
    comp_group = groups.get(comp_id)
    if not comp_group:
        print(f"🧵 {thread_id}: Group {comp_id} not found")
        return None
        
    comp_name = format_group_name(comp_group)
    comp_data = interpolated_groups.get(comp_id)
    
    if comp_data is None or comp_data.empty:
        print(f"🧵 {thread_id}: No cached interpolated data for {comp_name}")
        return None
    
    ref_binned  = bin_time_data(ref_data,  y_param, bin_minutes=15)
    comp_binned = bin_time_data(comp_data, y_param, bin_minutes=15)

    print(f"🧵 {thread_id}: Processing {comp_name} with {len(comp_data)} cached interpolated points")

    # Debug: Print some sample data to verify variance
    print(f"🧵 {thread_id}: Sample ref data variance: {ref_data[y_param].var():.6f}")
    print(f"🧵 {thread_id}: Sample comp data variance: {comp_data[y_param].var():.6f}")

    # Global comparison on per-well means
    ref_means = ref_data.groupby('well')[y_param].mean().values
    comp_means = comp_data.groupby('well')[y_param].mean().values
    
    print(f"🧵 {thread_id}: Ref means: {ref_means[:3]}... (variance: {ref_means.var():.6f})")
    print(f"🧵 {thread_id}: Comp means: {comp_means[:3]}... (variance: {comp_means.var():.6f})")
    
    overall_text = overall_color = ""

    if len(ref_means) < 2 or len(comp_means) < 2:
        overall_p = None
        overall_text = "Not enough wells for global test"
        overall_color = "warning"
    else:
        try:
            if test_type == 'ttest':
                _, overall_p = ttest_ind(ref_means, comp_means, equal_var=False)
            else:
                # Mann-Whitney U is non‐parametric two‐tailed by default
                _, overall_p = mannwhitneyu(ref_means, comp_means, alternative='two-sided')
            print(f"🧵 {thread_id}: Global t-test p-value: {overall_p}")
        except Exception as e:
            print(f"🧵 {thread_id}: Global t-test failed: {e}")
            overall_p = None
            overall_text = "Global test failed"
            overall_color = "warning"

    # ── Time-bin comparison (15 min windows) ──────────────────────────────
    ref_bins  = set(ref_binned['elapsed_bin_hours'])
    comp_bins = set(comp_binned['elapsed_bin_hours'])
    common_bins = sorted(ref_bins.intersection(comp_bins))

    if not common_bins:
        print(f"🧵 {thread_id}: No overlapping 15 min bins!")
        return None

    p_records = []
    for bin_hr in common_bins:
        ref_vals  = ref_binned [ref_binned ['elapsed_bin_hours'] == bin_hr][y_param].values
        comp_vals = comp_binned[comp_binned['elapsed_bin_hours'] == bin_hr][y_param].values
        
        if len(ref_vals) > 0 and len(comp_vals) > 0:
            try:
                if test_type == 'ttest':
                    _, p_val = ttest_ind(ref_vals, comp_vals, equal_var=False)
                else:
                    _, p_val = mannwhitneyu(ref_vals, comp_vals, alternative='two-sided')
            except Exception as e:
                print(f"🧵 {thread_id}: Test failed in bin {bin_hr}: {e}")
                p_val = np.nan
            
            p_records.append({
                'elapsed_hours': bin_hr,
                'p_value': p_val
            })

    p_df = pd.DataFrame(p_records)
    print(f"🧵 {thread_id}: Generated {len(p_df)} binned p-values")
        
    p_df = pd.DataFrame(p_records)
    print(f"🧵 {thread_id}: Generated {len(p_df)} p-values from cache, {p_df['p_value'].notna().sum()} non-NaN")
    
    # Debug: Print sample p-values
    sample_p_values = p_df['p_value'].dropna().head(5).tolist()
    print(f"🧵 {thread_id}: Sample p-values: {sample_p_values}")

    # Calculate significance statistics
    sig_mask = p_df['p_value'] < p_threshold
    sig_count = int(sig_mask.sum())
    
    total_tp = len(p_df)
    pct_sig = (sig_count / total_tp * 100) if total_tp > 0 else 0
    
    print(f"🧵 {thread_id}: Completed analysis for {comp_name}")
    
    # Return all the data needed for plotting and summary
    return {
        'comp_id': comp_id,
        'comp_name': comp_name,
        'color': color,
        'p_df': p_df,
        'ref_name': ref_name,
        'overall_p': overall_p,
        'overall_text': overall_text,
        'overall_color': overall_color,
        'sig_count': sig_count,
        'total_tp': total_tp,
        'pct_sig': pct_sig,
    }

@callback(
    Output('stats-results-output', 'children'),
    #
    Input('apply-filters_stats', 'n_clicks'),
    State('select_test', 'value'), 
    State('select-group-ref', 'value'),
    State('select-group-comp', 'value'),
    State('p_value_threshold', 'value'),
    State('well_groups_data', 'data'),
    State('selected_normalization', 'data'),
    State('selected_frequency', 'data'),
    prevent_initial_call=True
)
def perform_statistical_analysis(n_clicks, test_type, ref_group_id, comp_group_ids, p_threshold, well_groups_data, y_param, selected_freq):
    """
    Performs statistical analysis using cached interpolated per-minute data with multithreading.
    """
    # 1) Validate inputs
    if not n_clicks or not ref_group_id or not comp_group_ids or p_threshold is None:
        return dbc.Alert("Missing required inputs", color="warning")

    # 2) Check if we have cached interpolated data
    cached_interpolated = cache.get("interpolated_groups", [])
    if not cached_interpolated:
        return dbc.Alert("No cached interpolated data found. Please generate the graph first.", color="warning")

    # 3) Convert cached data to expected format
    interpolated_groups = get_interpolated_groups_from_cache(well_groups_data)
    
    # ⏱️ On filtre les interpolations si un t0 a été défini
    t0 = cache.get("t0")
    if t0 is not None:
        for group_id, df in interpolated_groups.items():
            if 'elapsed_hours' in df.columns:
                interpolated_groups[group_id] = df[df["elapsed_hours"] > 0].copy()
                
    if not interpolated_groups:
        return dbc.Alert("No interpolated data could be retrieved from cache", color="danger")

    # 4) Get reference group info
    groups = {g['id']: g for g in well_groups_data}
    ref_group = groups.get(ref_group_id)
    if not ref_group:
        return dbc.Alert("Reference group not found", color="danger")
    
    ref_name = format_group_name(ref_group)
    ref_data = interpolated_groups.get(ref_group_id)
    
    if ref_data is None or ref_data.empty:
        return dbc.Alert("No cached interpolated data for reference group", color="danger")

    print(f"🔍 Reference group has {len(ref_data)} cached interpolated points")

    # 5) Determine number of threads (CPU cores - 2, minimum 2)
    max_threads = max(2, os.cpu_count() - 2)
    actual_threads = min(max_threads, len(comp_group_ids))
    print(f"Using {actual_threads} threads (max available: {max_threads})")

    # 6) Prepare arguments for parallel processing
    palette = pcolors.qualitative.Set1
    thread_args = []
    
    for idx, comp_id in enumerate(comp_group_ids):
        color = palette[idx % len(palette)]
        args = (comp_id, groups, ref_data, ref_name, interpolated_groups, p_threshold, y_param, color, idx, test_type)
        thread_args.append(args)

    # 7) Run parallel analysis
    start_time = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=actual_threads) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_comparison_group, args): args for args in thread_args}
        
        # Collect results as they complete
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print(f"Thread generated an exception: {exc}")
                comp_id = args[0]
                print(f"Failed processing comparison group {comp_id}")

    elapsed_time = time.time() - start_time

    if not results:
        return dbc.Alert("No valid comparison results generated", color="danger")

    combined_graph, comparison_summaries = plot_stats(go, results, ref_name, p_threshold)
    
    # 10) Build results UI
    results_ui = [
        dbc.Card([
            dbc.CardHeader("Courbes de p-values"),
            dbc.CardBody([
                html.P(f"Analyse effectuée en {elapsed_time:.2f} secondes."),
                combined_graph
            ])
        ], className="mb-4")
    ]
    
    for s in comparison_summaries:
        # Fix the display logic to properly show p-values including 0.0
        overall_p_val = s['overall_p']
        if overall_p_val is None or (isinstance(overall_p_val, float) and np.isnan(overall_p_val)):
            overall_p_text = "P-value globale: N/A"
        else:
            overall_p_text = f"P-value globale: {overall_p_val:.6f}"
        
        results_ui.append(
            dbc.Card([
                dbc.CardHeader(f"{ref_name} vs {s['comp_name']}"),
                dbc.CardBody([
                    dbc.Alert(s['overall_text'], color=s['overall_color']) if (len(s.get('overall_text')) > 0) else None,
                    html.P(f"{overall_p_text} // Points significatifs: {s['sig_count']}/{s['total_tp']} ({s['pct_sig']:.1f}%)"),
                ])
            ], className="mb-3")
        )
    
    save_stats_info_to_file(
        test_type=test_type,
        ref_group_name=ref_name,
        comparison_summaries=comparison_summaries,
        p_threshold=p_threshold,
        y_param=y_param,
        selected_freq=selected_freq,
        t0=t0,
        elapsed_time=elapsed_time,
        groups=groups,
        comp_group_ids=comp_group_ids
    )

    return results_ui