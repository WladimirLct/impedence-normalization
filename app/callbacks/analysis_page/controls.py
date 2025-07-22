# File: callbacks/analysis_page/controls.py

from dash import callback, Output, Input, State, no_update
import pandas as pd

from config.cache import cache
from config.options import DEFAULT_FREQ

@callback(
    Output('select_norm', 'options'),
    Output('select_norm', 'value'),
    Output('select_frequency', 'options'),
    Output('select_frequency', 'value'),
    Output('select_t0', 'value'),
    #
    Input('url-p3', 'pathname'),
)
def populate_control_options(pathname):
    """Populate normalization and frequency options from data"""
    
    if pathname != '/analysis':
        return [], [], no_update, no_update, no_update
    
    raw_data = cache.get("data")
    if raw_data is None or raw_data.empty:
        return [], [], no_update, no_update, no_update
    
    # Normalization options based on available columns
    norm_columns = ['absz', 'absz_t0', 'absz_max', 'absz_min_max']
    available_norms = [col for col in norm_columns if col in raw_data.columns]
    
    norm_options = [
        {'label': 'Impédance brute', 'value': 'absz'},
        {'label': 'Impédance normalisée T0', 'value': 'absz_t0'},
        {'label': 'Impédance normalisée max', 'value': 'absz_max'},
        {'label': 'Impédance min-max', 'value': 'absz_min_max'}
    ]
    
    # Filter to only available options
    norm_options = [opt for opt in norm_options if opt['value'] in available_norms]
    
    def_freq_found = False
    # Frequency options from unique frequencies in data
    if 'frequency' in raw_data.columns:
        unique_frequencies = sorted(raw_data['frequency'].dropna().unique())
        freq_options = [{'label': f'{freq} Hz', 'value': freq} for freq in unique_frequencies]
        if DEFAULT_FREQ in unique_frequencies: def_freq_found = True
    else:
        freq_options = []
    
    def_freq = DEFAULT_FREQ if def_freq_found else freq_options[0]['value']

    t0 = cache.get("t0")
    if t0 is None:
        t0 = 0

    return norm_options, norm_options[0]['value'], freq_options, def_freq, t0

@callback(
    Output('selected_normalization', 'data'),
    Output('selected_frequency', 'data'),
    Output('std_bar_frequency', 'data'),
    #
    Input('apply-filters', 'n_clicks'),
    State('select_norm', 'value'),
    State('select_frequency', 'value'),
    State('std-bar-frequency', 'value'),
    State('select_t0', 'value'),
    prevent_initial_call=True,
)
def store_selected_parameters(n_clicks, selected_norm, selected_freq, std_freq, select_t0):
    if not n_clicks:
        return no_update, no_update, no_update

    if std_freq < 0:
        std_freq = 10

    raw_data = cache.get("data")
    if raw_data is not None and not raw_data.empty:
        date_min = raw_data["date"].min()
        
        # Try to convert select_t0 to float, default to 0 if it fails
        try:
            t0_hours = float(select_t0)
        except (ValueError, TypeError):
            t0_hours = 0.0
        
        t0 = date_min + pd.to_timedelta(t0_hours, unit='h')
        cache["t0"] = t0

    return selected_norm, selected_freq, std_freq
