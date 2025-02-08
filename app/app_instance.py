# app_instance.py

import dash
from dash import Dash
import dash_bootstrap_components as dbc
import diskcache
from dash.long_callback import DiskcacheLongCallbackManager  # Correct import
import pandas as pd

# Clear cache folder for fresh start
import shutil
shutil.rmtree("./cache", ignore_errors=True)

# # Initialize Diskcache
# cache = diskcache.Cache("./cache")  # Ensure this directory exists
# long_callback_manager = DiskcacheLongCallbackManager(cache)
# # Initialize Diskcache
# cache = diskcache.Cache("./cache")
# background_callback_manager = dash.long_callback.DiskcacheLongCallbackManager(cache)
# app.long_callback_manager = background_callback_manager

# Initialize Dash application with the long_callback_manager
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    # long_callback_manager=long_callback_manager,  # Pass manager here
)
app.title = "BioImpedance Analyzer"

data_cache = {}  # Store data in memory

def load_data(session_id, file_path):
    if session_id in data_cache:
        # Data is already cached
        df = data_cache[session_id]
    else:
        # Read from the file and store in the cache
        df = pd.read_csv(file_path)
        data_cache[session_id] = df
    return df