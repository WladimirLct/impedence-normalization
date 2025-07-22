# File: utils/graph.py

import plotly.graph_objs as go
import plotly.express as px
import plotly.colors as pc

import numpy as np
import pandas as pd
from config.cache import cache

from datetime import datetime

from utils.normalizer import normalize_t0
COLOR_CYCLE = px.colors.qualitative.Plotly

def convert_date_to_elapsed_hours(data, date_column):
    """Convert datetime column to elapsed hours from the earliest timestamp."""
    if date_column not in data.columns:
        return data
    
    # Ensure datetime format
    data = data.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Find the earliest timestamp across all data
    start_time = data[date_column].min()
    
    # Calculate elapsed hours as float
    data['elapsed_hours'] = (data[date_column] - start_time).dt.total_seconds() / 3600
    
    return data

def plot_graph(data, groups_data, y_param='absz', x_param='date', std_bar_freq=10):
    """Plot aggregated per-minute curves and standard-deviation bars for each group."""
    
    if data is None or data.empty:
        return {'data': [], 'layout': {'title': 'Aucune donnée'}}
    
    # Convert dates to elapsed hours if using date parameter
    use_elapsed_time = False
    if x_param == 'date' and x_param in data.columns:
        data = convert_date_to_elapsed_hours(data, x_param)
        x_param = 'elapsed_hours'  # Switch to using elapsed hours
        use_elapsed_time = True
    
    if y_param == 'absz_t0':
        data = normalize_t0(data)
        print("use normalized t0")
    elif y_param not in data.columns:
        return {'data': [], 'layout': {'title': f'Paramètre {y_param} non trouvé'}}

    # Initialize interpolated groups cache
    interpolated_groups = []
    
    fig = go.Figure()
    for i, group in enumerate(groups_data or []):
        wells = group.get('wells')
        concentration = group.get('concentration')
        unit = group.get('unit')
        if not wells:
            continue
        
        # Get color for this group
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        
        group_df = data[data['well'].isin(wells)][[x_param, 'well', y_param]].copy()
        if group_df.empty:
            continue
        
        if len(wells) > 1:
            # Compute both group stats and individual well interpolated data
            full_stats, interpolated_wells = compute_group_per_minute_stats(group_df, x_param, y_param, use_elapsed_time)

            # Save both aggregated and individual well interpolated data to cache
            interpolated_group = {
                "id": group['id'],
                "name": group['name'],
                "concentration": concentration,
                "unit": unit,
                "label": group.get('label', ''),
                "interpolated_avg": full_stats,
                "interpolated_wells": interpolated_wells  # Individual well data for t-tests
            }
            interpolated_groups.append(interpolated_group)

            # plot the mean-curve with explicit color
            fig.add_trace(go.Scatter(
                x=full_stats[x_param],
                y=full_stats['mean'],
                mode='lines',
                name=f"{group['name']} ({concentration} {unit})", # moyenne/min",
                line=dict(color=color, width=3),  # Set explicit color
                legendgroup=f"group_{group['name']}{concentration}{unit}",
                hovertemplate=(
                    f"<b>{group['name']}</b><br>"
                    f"{'Heures écoulées' if use_elapsed_time else x_param}: %{{x:.2f}}<br>"
                    f"{y_param}: %{{y:.3f}}<extra></extra>"
                )
            ))

            if std_bar_freq > 0:
                # subsample for σ-bars
                if std_bar_freq and len(full_stats) > std_bar_freq:
                    step = max(1, len(full_stats) // std_bar_freq)
                    err_stats = full_stats.iloc[::step]
                else:
                    err_stats = full_stats

                # Create semi-transparent version of the color for error bars
                try:
                    # Handle both hex and named colors
                    if color.startswith('#'):
                        rgb = pc.hex_to_rgb(color)
                    else:
                        # For named colors, convert to hex first
                        hex_color = pc.convert_colors_to_same_type([color], colortype='hex')[0][0]
                        rgb = pc.hex_to_rgb(hex_color)
                    error_color = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.3)"
                except:
                    # Fallback to original color if conversion fails
                    error_color = color

                fig.add_trace(go.Scatter(
                    x=err_stats[x_param],
                    y=err_stats['mean'],
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    legendgroup=f"group_{group['name']}{concentration}{unit}",
                    error_y=dict(
                        type='data', 
                        array=err_stats['std'],
                        visible=True, 
                        color=error_color,  # Use matching color with transparency
                        thickness=2, 
                        width=4
                    ),
                    showlegend=False
                ))
        else:
            well = wells[0]
            single = group_df[group_df['well'] == well].sort_values(x_param)
            fig.add_trace(go.Scatter(
                x=single[x_param],
                y=single[y_param],
                mode='lines',
                name=f"{group['name']}",
                line=dict(color=color, width=2),  # Set explicit color for single wells too
                hovertemplate=(
                    f"<b>{well}</b><br>"
                    f"{'Heures écoulées' if use_elapsed_time else x_param}: %{{x:.2f}}<br>"
                    f"{y_param}: %{{y:.3f}}<extra></extra>"
                )
            ))

    # Save interpolated groups to cache
    cache["interpolated_groups"] = interpolated_groups

    # Set appropriate x-axis title
    x_axis_title = 'Elapsed time (h)' if use_elapsed_time else 'Date et heure'
    
    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=get_y_label(y_param),
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=50, r=150, t=50, b=50)
    )

    # fig.write_image('tmp/main_graph.pdf', width=1200, height=800)

    save_graph_info_to_file(
        data=data,
        groups_data=groups_data,
        y_param=y_param,
        x_param=x_param,
        std_bar_freq=std_bar_freq,
        use_elapsed_time=use_elapsed_time,
        interpolated_groups=interpolated_groups
    )

    return fig

def get_y_label(y_param):
    labels = {
        'absz': 'Zabs (Ohms)',
        'absz_t0': 'Zabs normalized T0',
        'absz_max': 'Zabs normalized (Max)',
        'absz_min_max': 'Zabs normalized (Min-Max)'
    }
    return labels.get(y_param, y_param)

def save_graph_info_to_file(data, groups_data, y_param, x_param, std_bar_freq, use_elapsed_time, interpolated_groups):
    """
    Save all main graph information to tmp/graph_info.txt
    """
    # Get t0 from cache if available
    t0 = cache.get("t0")
    
    # Calculate data statistics
    total_data_points = len(data) if data is not None else 0
    unique_wells = data['well'].nunique() if data is not None and 'well' in data.columns else 0
    
    # Time range information
    time_range_info = ""
    if data is not None and not data.empty:
        if use_elapsed_time and 'elapsed_hours' in data.columns:
            min_time = data['elapsed_hours'].min()
            max_time = data['elapsed_hours'].max()
            time_range_info = f"Time range: {min_time:.2f} to {max_time:.2f} hours"
        elif 'date' in data.columns:
            min_date = data['date'].min()
            max_date = data['date'].max()
            time_range_info = f"Date range: {min_date} to {max_date}"
    
    # Write to file
    with open('tmp/graph_info.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MAIN GRAPH VISUALIZATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Graph Parameters
        f.write("GRAPH PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Y Parameter: {y_param} ({get_y_label(y_param)})\n")
        f.write(f"X Parameter: {x_param}\n")
        f.write(f"Using Elapsed Time: {use_elapsed_time}\n")
        f.write(f"Standard Deviation Bar Frequency: {std_bar_freq}\n")
        f.write(f"T0 (start time): {t0 if t0 is not None else 'Not set'}\n\n")
        
        # Data Overview
        f.write("DATA OVERVIEW:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Data Points: {total_data_points:,}\n")
        f.write(f"Unique Wells: {unique_wells}\n")
        f.write(f"{time_range_info}\n")
        
        if data is not None and not data.empty:
            # Y parameter statistics
            y_stats = data[y_param].describe()
            f.write(f"\n{y_param} Statistics:\n")
            f.write(f"  Mean: {y_stats['mean']:.4f}\n")
            f.write(f"  Std Dev: {y_stats['std']:.4f}\n")
            f.write(f"  Min: {y_stats['min']:.4f}\n")
            f.write(f"  Max: {y_stats['max']:.4f}\n")
        
        f.write("\n")
        
        # Group Information
        f.write("GROUP INFORMATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Groups: {len(groups_data) if groups_data else 0}\n\n")
        
        if groups_data:
            for i, group in enumerate(groups_data, 1):
                wells = group.get('wells', [])
                concentration = group.get('concentration', 'N/A')
                unit = group.get('unit', '')
                name = group.get('name', f'Group {i}')
                
                f.write(f"{i}. {name}\n")
                f.write(f"   Concentration: {concentration} {unit}\n")
                f.write(f"   Wells: {', '.join(wells)} ({len(wells)} wells)\n")
                
                # Group-specific data statistics
                if data is not None and wells:
                    group_data = data[data['well'].isin(wells)]
                    if not group_data.empty:
                        group_points = len(group_data)
                        group_y_mean = group_data[y_param].mean()
                        group_y_std = group_data[y_param].std()
                        f.write(f"   Data Points: {group_points:,}\n")
                        f.write(f"   {y_param} Mean: {group_y_mean:.4f} ± {group_y_std:.4f}\n")
                
                f.write("\n")
        
        # Interpolated Data Information
        f.write("INTERPOLATED DATA:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Groups with Interpolated Data: {len(interpolated_groups)}\n\n")
        
        for interp_group in interpolated_groups:
            name = interp_group.get('name', 'Unknown')
            concentration = interp_group.get('concentration', 'N/A')
            unit = interp_group.get('unit', '')
            
            f.write(f"- {name} ({concentration} {unit})\n")
            
            # Interpolated average data info
            avg_data = interp_group.get('interpolated_avg')
            if avg_data is not None and not avg_data.empty:
                f.write(f"  Interpolated Points: {len(avg_data)}\n")
                f.write(f"  Mean Range: {avg_data['mean'].min():.4f} to {avg_data['mean'].max():.4f}\n")
                f.write(f"  Std Dev Range: {avg_data['std'].min():.4f} to {avg_data['std'].max():.4f}\n")
            
            # Individual wells interpolated data info
            wells_data = interp_group.get('interpolated_wells', [])
            if len(wells_data) > 0:
                f.write(f"  Individual Wells Interpolated: {len(wells_data)}\n")
                for well_id, well_df in wells_data.items():
                    if well_df is not None and not well_df.empty:
                        f.write(f"    {well_id}: {len(well_df)} points\n")
            
            f.write("\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")
    
    print(f"📊 Graph visualization info saved to tmp/graph_info.txt")


def compute_group_per_minute_stats(group_df, x_param, y_param, use_elapsed_time=False):
    """
    Pivot each well into its own column, resample at 1-minute intervals
    (with linear interpolation), then compute mean & std across wells.
    Removes initial rows where only a single curve contributes to the statistics.
    Returns both group stats and individual well interpolated data.
    """
    wide = group_df.pivot_table(index=x_param, columns='well', values=y_param)
    
    if use_elapsed_time:
        # For elapsed hours, we need to create a proper time-based index for resampling
        start_time = pd.Timestamp('2000-01-01')  # Arbitrary start date for resampling
        time_index = start_time + pd.to_timedelta(wide.index, unit='h')
        wide.index = time_index
        
        wide_min = (
            wide
            .resample('1min')
            .mean()                       
            .interpolate(method='time')
        )
        
        # Remove rows where only a single curve contributes
        valid_rows = wide_min.notna().sum(axis=1) > 1
        wide_min = wide_min[valid_rows]
        
        # Convert back to elapsed hours
        elapsed_hours = (wide_min.index - start_time).total_seconds() / 3600
        
        # Group statistics
        group_stats = pd.DataFrame({
            x_param: elapsed_hours,
            'mean': wide_min.mean(axis=1),
            'std':  wide_min.std(axis=1, ddof=0)
        })
        
        # Individual well data for t-tests (convert to long format)
        well_data_list = []
        for well in wide_min.columns:
            well_df = pd.DataFrame({
                x_param: elapsed_hours,
                'well': well,
                y_param: wide_min[well].values
            })
            well_data_list.append(well_df)
        
        interpolated_wells = pd.concat(well_data_list, ignore_index=True)
        
        return group_stats, interpolated_wells
    else:
        wide_min = (
            wide
            .resample('1min')
            .mean()                       
            .interpolate(method='time')
        )
        
        # Remove rows where only a single curve contributes
        valid_rows = wide_min.notna().sum(axis=1) > 1
        wide_min = wide_min[valid_rows]

        # Group statistics
        group_stats = pd.DataFrame({
            x_param: wide_min.index,
            'mean': wide_min.mean(axis=1),
            'std':  wide_min.std(axis=1, ddof=0)
        })
        
        # Individual well data for t-tests (convert to long format)
        well_data_list = []
        for well in wide_min.columns:
            well_df = pd.DataFrame({
                x_param: wide_min.index,
                'well': well,
                y_param: wide_min[well].values
            })
            well_data_list.append(well_df)
        
        interpolated_wells = pd.concat(well_data_list, ignore_index=True)
        
        return group_stats, interpolated_wells