# File: utils/stats.py

import numpy as np
from datetime import datetime

from dash import dcc
from utils.tools import format_group_name

def bin_time_data(df, y_param, bin_minutes=15):
    """
    Bins the DataFrame into windows of bin_minutes (default 15),
    averaging y_param per well in each window. Returns a DataFrame
    with columns ['well','elapsed_bin_hours', y_param].
    """
    # 1) Copy and convert hours → integer minutes
    df = df.copy()
    df['elapsed_minutes'] = (df['elapsed_hours'] * 60).round().astype(int)
    
    # 2) Floor into bins of size bin_minutes
    df['bin_start'] = (df['elapsed_minutes'] // bin_minutes) * bin_minutes
    
    # 3) Convert bin back to hours for labeling
    df['elapsed_bin_hours'] = df['bin_start'] / 60.0
    
    # 4) Average y_param per well within each bin
    binned = (
        df
        .groupby(['well', 'elapsed_bin_hours'])[y_param]
        .mean()
        .reset_index()
    )
    return binned

def plot_stats(go, results, ref_name, p_threshold=0.05):
    # 8) Build the plot from results
    p_value_fig = go.Figure()
    comparison_summaries = []

    for result in results:
        # Add p-value trace
        p_value_fig.add_trace(go.Scatter(
            x=result['p_df']['elapsed_hours'],
            y=result['p_df']['p_value'],
            mode='lines+markers',
            name=f"{result['ref_name']} vs {result['comp_name']}",
            line=dict(color=result['color'], width=3),
            marker=dict(color=result['color'], size=6),
            connectgaps=True
        ))
        
        # Add to summary
        comparison_summaries.append({
            'comp_name': result['comp_name'],
            'overall_p': result['overall_p'],
            'overall_text': result['overall_text'],
            'overall_color': result['overall_color'],
            'sig_count': result['sig_count'],
            'total_tp': result['total_tp'],
            'pct_sig': result['pct_sig']
        })

    print(f"🔍 Added {len(p_value_fig.data)} traces to plot using cached data")

    # 9) Finalize plot
    p_value_fig.add_hline(
        y=p_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold = {p_threshold}",
        annotation_position="bottom right"
    )
    
    p_value_fig.update_layout(
        title=f"P-values over time vs {ref_name}",
        xaxis_title="Elapsed Hours",
        yaxis_title="P-value",
        yaxis=dict(range=[0, 1]),
        height=500,
        showlegend=True
    )
    
    # p_value_fig.write_image('tmp/stat_graph.pdf', width=1200, height=500)

    combined_graph = dcc.Graph(figure=p_value_fig, style={'height': '500px'})
    combined_graph.id = 'stats-graph'

    return combined_graph, comparison_summaries

def save_stats_info_to_file(test_type, ref_group_name, comparison_summaries, p_threshold, y_param, selected_freq, t0, elapsed_time, groups, comp_group_ids):
    """
    Save all statistical analysis information to tmp/stats_info.txt
    """
    # Get comparison group names
    comp_group_names = []
    for comp_id in comp_group_ids:
        comp_group = groups.get(comp_id)
        if comp_group:
            comp_group_names.append(format_group_name(comp_group))
    
    # Write to file
    with open('tmp/stats_info.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis duration: {elapsed_time:.2f} seconds\n\n")
        
        # Analysis Parameters
        f.write("ANALYSIS PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Type: {test_type}\n")
        f.write(f"Reference Group: {ref_group_name}\n")
        f.write(f"Comparison Groups: {', '.join(comp_group_names)}\n")
        f.write(f"P-value Threshold: {p_threshold}\n")
        f.write(f"Normalization Parameter: {y_param}\n")
        f.write(f"Selected Frequency: {selected_freq}\n")
        f.write(f"T0 (start time): {t0 if t0 is not None else 'Not set'}\n\n")
        
        # Group Details
        f.write("GROUP DETAILS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Reference Group: {ref_group_name}\n")
        ref_group = next((g for g in groups.values() if format_group_name(g) == ref_group_name), None)
        if ref_group:
            f.write(f"  - Wells: {', '.join(ref_group.get('wells', []))}\n")
            f.write(f"  - Concentration: {ref_group.get('concentration', 'N/A')}\n")
        
        f.write("\nComparison Groups:\n")
        for comp_id in comp_group_ids:
            comp_group = groups.get(comp_id)
            if comp_group:
                comp_name = format_group_name(comp_group)
                f.write(f"  - {comp_name}\n")
                f.write(f"    Wells: {', '.join(comp_group.get('wells', []))}\n")
                f.write(f"    Concentration: {comp_group.get('concentration', 'N/A')}\n")
        
        f.write("\n")
        
        # Statistical Results
        f.write("STATISTICAL RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for i, summary in enumerate(comparison_summaries, 1):
            f.write(f"\n{i}. {ref_group_name} vs {summary['comp_name']}\n")
            f.write("   " + "=" * 40 + "\n")
            
            # Overall p-value
            overall_p_val = summary['overall_p']
            if overall_p_val is None or (isinstance(overall_p_val, float) and np.isnan(overall_p_val)):
                f.write("   Overall P-value: N/A\n")
            else:
                f.write(f"   Overall P-value: {overall_p_val:.6f}\n")
            
            # Significance summary
            f.write(f"   Significant Points: {summary['sig_count']}/{summary['total_tp']} ({summary['pct_sig']:.1f}%)\n")
            
            # Overall interpretation
            if summary.get('overall_text'):
                f.write(f"   Interpretation: {summary['overall_text']}\n")
            
            f.write(f"   Statistical Significance: {'Yes' if summary['overall_color'] == 'success' else 'No'}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")
    
    print(f"📄 Statistical analysis info saved to tmp/stats_info.txt")
