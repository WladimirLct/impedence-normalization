# main_page.py

import os
import pandas as pd
import plotly
import plotly.graph_objects as go

from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import dash.exceptions
import dash

from app_instance import app, data_cache  # Import both app and cache from app_instance
import uuid

from io import StringIO
import time

# Local imports
from utils import load_wells

# Configuration
CONFIG = {
    "impedance_params": ['Param0', 'Param1', 'AbsZ', 'RealZ', 'ImagZ', 'PhaseZ'],
    "group_size": 3,
    "date_format": '%Y-%m-%d %H:%M:%S',
    "save_dir": "./analysis_results",
    "colorscale": plotly.colors.qualitative.Dark24
}

# Create the save directory if it doesn't exist
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# Main page layout
main_layout = dbc.Container([ 
    dcc.Store(id='group-assignments', storage_type='session'),

    dbc.Row([ 
        # Left Column - Data Management
        dbc.Col([ 
            # Data Input Card
            dbc.Card([ 
                dbc.CardHeader("Data Management", className="h5"), 
                dbc.CardBody([ 
                    dbc.InputGroup([ 
                        dbc.Input( 
                            id="input-path", 
                            placeholder="Enter/paste directory path", 
                            className="rounded-start" 
                        ), 
                        dbc.Button( 
                            "Analyze", 
                            id="analyze-button", 
                            color="primary", 
                            className="rounded-end" 
                        ), 
                    ], className="mb-3"), 

                    html.Small( 
                        id="loading-output", 
                        className="text-muted d-block text-center", 
                        children="Ready"  # Initialize with 'Ready'
                    ) 
                ]) 
            ], className="mb-4"),

            # Data Export Card
            dbc.Card([ 
                dbc.CardHeader("Data Export", className="h5"), 
                dbc.CardBody([ 
                    dbc.Row([ 
                        dbc.Col([ 
                            dbc.Label("Frequencies:", className="mb-2"), 
                            dcc.Dropdown( 
                                id="download-frequency-selector", 
                                multi=True, 
                                placeholder="Select frequencies...", 
                                className="dropdown-primary" 
                            ) 
                        ], md=6),

                        dbc.Col([ 
                            dbc.Label("Wells:", className="mb-2"), 
                            dcc.Dropdown( 
                                id="download-well-selector", 
                                multi=True, 
                                placeholder="Select wells...", 
                                className="dropdown-primary" 
                            ) 
                        ], md=6)
                    ], className="g-2 mb-3"),

                    dbc.Button( 
                        "Download CSV Bundle", 
                        id="download-button", 
                        color="success", 
                        className="w-100", 
                        outline=True 
                    ),
                    dcc.Download(id="download-csv"),

                    dbc.Button( 
                        "Statistical Analysis", 
                        id="nav-to-stats", 
                        color="secondary", 
                        className="w-100 mt-3", 
                        outline=True 
                    ),
                ])
            ]) 
        ], md=3, className="pe-3"),

        # Right Column - Visualization & Controls
        dbc.Col([ 
            # Control Panel Card
            dbc.Card([ 
                dbc.CardHeader("Visualization Controls", className="h5"), 
                dbc.CardBody([ 
                    dbc.Row([ 
                        # Normalization Controls
                        dbc.Col([ 
                            dbc.Label("Normalization Method:", className="mb-2"), 
                            dcc.Dropdown( 
                                id="normalization-method", 
                                options=[ 
                                    {'label': 'Absolute Z', 'value': 'AbsZ'}, 
                                    {'label': 'T0 Normalized', 'value': 'AbsZ_t0'}, 
                                    {'label': 'Max Normalized', 'value': 'AbsZ_max'}, 
                                    {'label': 'Min-Max Scaled', 'value': 'AbsZ_min_max'}, 
                                    {'label': 'Control-Normalized', 'value': 'AbsZ_control'}, 
                                    {'label': 'Differential', 'value': 'AbsZ_diff'}
                                ], 
                                value='AbsZ', 
                                clearable=False, 
                                className="mb-3"
                            ) 
                        ], md=4),

                        # Frequency Selection
                        dbc.Col([ 
                            dbc.Label("Active Frequencies:", className="mb-2"), 
                            dcc.Dropdown( 
                                id="frequency-selector", 
                                multi=True, 
                                clearable=False, 
                                className="mb-3"
                            ) 
                        ], md=4),

                        # Display Options
                        dbc.Col([ 
                            dbc.Label("Display Settings:", className="mb-2"), 
                            dbc.Checklist( 
                                options=[ 
                                    {"label": " Group Wells", "value": "group-wells"} 
                                ], 
                                value=[], 
                                id="view-options", 
                                switch=True, 
                                className="mb-3"
                            ), 
                            dbc.Label("Data Density:", className="mb-2"), 
                            dcc.Slider( 
                                id="data-resolution", 
                                min=1, 
                                max=100, 
                                step=1, 
                                value=1, 
                                marks=None, 
                                tooltip={"placement": "bottom", "always_visible": True} 
                            ) 
                        ], md=4),

                        dbc.Col( 
                            dbc.Label("Std Multiplier:"), 
                            width="auto" 
                        ), 
                        dbc.Col( 
                            dcc.Slider( 
                                id="std-scale", 
                                min=0, 
                                max=3.0, 
                                step=0.1, 
                                value=1.0, 
                                tooltip={"placement": "bottom"} 
                            ), 
                            className="ps-0" 
                        ) 
                    ], className="g-3") 
                ]) 
            ], className="mb-4"),

            # Main Visualization
            dbc.Card([ 
                dbc.CardBody([ 
                    dcc.Graph( 
                        id="impedance-plot", 
                        config={"displayModeBar": True}, 
                        style={ 
                            "height": "65vh", 
                            "minHeight": "500px" 
                        }, 
                        className="border rounded-3" 
                    ) 
                ]) 
            ]),

            # Comment Section Card
            dbc.Card([ 
                dbc.CardHeader("Comment", className="h5"), 
                dbc.CardBody([
                    dcc.Store(id="comment-store", storage_type="session"),  # Ajout du dcc.Store ici 
                    dbc.Textarea( 
                        id="comment-textarea", 
                        placeholder="Enter your comments here...", 
                        style={"height": "150px"} 
                    ),
                    dbc.Button("Save", id="save-button", color="primary", className="mt-3")  # Ajout du bouton Save 
                ]) 
            ], className="mt-4"),  # Adding margin-top for spacing
        ], md=9, className="ps-3")
    ], className="g-4"),

    # Hidden elements for callbacks
    html.Div(id="dummy-output", style={"display": "none"})
], fluid=True, className="dbc bg-light")


# Define callbacks in a function to be registered in app.py
def register_main_callbacks(app):
        
    # Use callback for data loading and processing
    @app.callback(
        Output("frequency-selector", "options"),
        Output("frequency-selector", "value"),
        Output("loading-output", "children"),
        Output("session-store", "data"),
        Input("analyze-button", "n_clicks"),
        State("input-path", "value"),
        running=[
            (Output("analyze-button", "disabled"), True, False),
            (Output("loading-output", "children"), "Processing...", "Ready"),
        ],
        prevent_initial_call=True
    )
    def load_and_process_data(n_clicks, path):

        # Prevent callback execution if the button hasn't been clicked
        if not n_clicks or n_clicks == 0:
            raise dash.exceptions.PreventUpdate

        # Validate the input path
        if not path or not os.path.exists(path):
            return dash.no_update, dash.no_update, "Invalid directory path", dash.no_update

        # Generate a unique session ID
        session_id = str(uuid.uuid4())

        # ** Clear the data_cache to remove old data **
        data_cache.clear()        

        try:
            # Check if processed file already exists
            save_path = os.path.join(CONFIG["save_dir"], f"{os.path.basename(path)}.csv")

            if os.path.exists(save_path):
                # If file exists, just read it
                df = pd.read_csv(save_path)
            else:
                # If not, process and save it
                df = load_wells(path, save_path, CONFIG["impedance_params"])
                df.to_csv(save_path, index=False)

            # Store the dataframe in Diskcache using session ID
            data_cache[session_id] = df

            freq_options = [{'label': f"{frq:.2f} Hz", 'value': frq}
                            for frq in sorted(df['Frequency'].unique())]
            default_freq = [df['Frequency'].iloc[0]]

            return freq_options, default_freq, "Data loaded successfully", session_id

        except Exception as e:
            return dash.no_update, dash.no_update, f"Error: {str(e)}", dash.no_update

    # Callback to update visualization
     # Modify update_visualization callback to retrieve data from Diskcache
    @app.callback(
        [Output("impedance-plot", "figure"),
         Output("group-assignments", "data")],
        [Input("session-store", "data"),
         Input("normalization-method", "value"),
         Input("frequency-selector", "value"),
         Input("view-options", "value"),
         Input("std-scale", "value"),
         Input("data-resolution", "value")],
        [State("input-path", "value"),
         State("group-assignments", "data")]
    )
    def update_visualization(session_id, norm_method, frequencies, view_options, std_scale, data_resolution, path, groups):
        if not session_id or not frequencies:
            return go.Figure(layout={'template': 'plotly_white'}), dash.no_update

        df = data_cache.get(session_id)
        if df is None:
            return go.Figure(layout={'template': 'plotly_white'}), dash.no_update
        filtered_df = filter_frequencies(df, frequencies)

        # Handle grouping and averaging if enabled
        if 'group-wells' in view_options:
            if not groups:
                groups = create_well_groups(filtered_df)
            filtered_df = apply_group_averaging(filtered_df, groups, norm_method)
            return generate_plot(filtered_df, norm_method, std_scale, data_resolution), groups
        else:
            return generate_plot(filtered_df, norm_method, std_scale, data_resolution), dash.no_update
    
    # Callback to populate download dropdowns
    @app.callback(
        [Output("download-frequency-selector", "options"),
        Output("download-well-selector", "options")],
        Input("session-store", "data")
    )
    def update_download_options(session_id):
        if not session_id:
            return [], []

        df = data_cache.get(session_id)
        if df is None:
            return [], []

        freq_options = [{'label': f"{frq:.2f} Hz", 'value': frq} for frq in sorted(df['Frequency'].unique())]
        well_options = [{'label': w, 'value': w} for w in sorted(df['Well'].unique())]
        return freq_options, well_options

    # Download callback
    @app.callback(
        Output("download-csv", "data"),
        Input("download-button", "n_clicks"),
        [State("session-store", "data"),
        State("download-frequency-selector", "value"),
        State("download-well-selector", "value")],
        prevent_initial_call=True
    )
    def generate_csv(n_clicks, session_id, frequencies, wells):
        if not session_id or not frequencies or not wells:
            raise dash.exceptions.PreventUpdate

        df = data_cache.get(session_id)
        if df is None:
            raise dash.exceptions.PreventUpdate

        filtered_df = df[
            df['Frequency'].isin(frequencies) &
            df['Well'].isin(wells)
        ]

        if filtered_df.empty:
            raise dash.exceptions.PreventUpdate

        return dcc.send_data_frame(
            filtered_df.to_csv,
            filename=f"bioimpedance_data_{time.strftime('%Y%m%d-%H%M%S')}.csv",
            index=False
        )
    
    #Save comment
    @app.callback(
        Output("comment-store", "value"),
        [Input("save-button", "n_clicks")],
        [State("comment-textarea", "value")],
        prevent_initial_call=True
    )
    def save_comment(n_clicks, comment):
        try:
            print(f"n_clicks: {n_clicks}, Comment: {comment}")
            if n_clicks and comment:
                print(comment)
                return {"comment": comment}
            return dash.no_update
        except Exception as e:
            print(f"Error: {e}")

# Helper Functions
def filter_frequencies(df, frequencies):
    return df[df['Frequency'].isin(frequencies)]

def create_well_groups(df):
    wells = sorted(df["Well"].unique())
    # Group wells based on group size
    grouped_wells = [wells[i:i + CONFIG["group_size"]] for i in range(0, len(wells), CONFIG["group_size"])]
    return grouped_wells

def apply_group_averaging(df, groups, norm_method):
    averaged_data = []
    for group in groups:
        group_df = df[df['Well'].isin(group)]
        group_mean = group_df.groupby(['hours', 'Frequency']).mean().reset_index()
        group_mean['Well'] = f"Group {', '.join(group)}"
        averaged_data.append(group_mean)
    return pd.concat(averaged_data, ignore_index=True)

def prepare_error_bars(view_options, std_scale):
    return {
        'std_scale': std_scale
    }

def generate_plot(df, norm_method, std_scale=1.0, data_resolution=1):
    fig = go.Figure()
    colors = plotly.colors.qualitative.Dark24

    for idx, well in enumerate(df['Well'].unique()):
        well_data = df[df['Well'] == well].sort_values('hours')

        # Apply data resolution
        if data_resolution > 1:
            well_data = well_data.iloc[::data_resolution]

        color = colors[idx % len(colors)]

        # Add continuous line for each well
        fig.add_trace(go.Scatter(
            x=well_data['hours'],
            y=well_data[norm_method],
            name=well,
            mode='lines',  # Changed from 'lines+markers' to 'lines'
            line=dict(width=2, color=color),
        ))

        # Add standard deviation area
        std = well_data[norm_method].std() * std_scale
        upper_bound = well_data[norm_method] + std
        lower_bound = well_data[norm_method] - std

        # Convert hex to rgba for fill color
        hex_color = color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        fillcolor = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)'

        # Add upper and lower bounds for std area
        fig.add_trace(go.Scatter(
            x=well_data['hours'],
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=well_data['hours'],
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=fillcolor,
            showlegend=False
        ))

    fig.update_layout(
        template='plotly_white',
        title=f'Impedance Over Time ({norm_method})',
        xaxis_title='Time (hours)',
        yaxis_title=f'Impedance ({norm_method})',
        hovermode='closest',
        legend_title='Wells',
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

    return fig