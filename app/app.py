import os, math
import pandas
from datetime import datetime
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import webbrowser

import dash
from dash import set_props, DiskcacheManager, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq

import utils

import diskcache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# Impedance Parameters to include; set to None to include all available parameters
impedance_parameters = ['Param0', 'Param1', 'AbsZ', 'RealZ', 'ImagZ', 'PhaseZ']

# direcory to save CSV
SAVE_DIRECTORY = "./saved"

# number of well per group
GROUPS_SIZE = 3

# format for date hours
fmt = '%Y-%m-%d %H:%M:%S'

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], background_callback_manager=background_callback_manager)
app.layout = html.Div(
    [
        dcc.Store(id='path'),
        dcc.Store(id='groups'),
        html.H1("File Browser"),
        html.H2("Set path"),
        html.Div([
            dcc.Input(id="input-path", type="text", debounce=True, style={'width' : '60%'}),
            dbc.Button("Confirmation", id="confirmation-button", n_clicks=0),
            html.Div([
                dbc.Spinner(id="spinner", size="md"),
            ], id="spinner-container", style={'display': 'none'}),
        ]),

        html.Div([
            html.H3("Collecting measurements..."),
            dbc.Progress(id="animated-progress-bar", style={'width' : '70%'}),
        ], id="animated-progress-bar-container", style={'display': 'none'}),

        html.Div([
            html.H3("Normalizing data..."),
            dbc.Progress(id="process-progress-bar", style={'width' : '70%'}),
        ], id="process-progress-bar-container", style={'display': 'none'}),

        html.Div([
            dcc.Dropdown([], [], id="frequency-dropdown", multi=True, style={'width' : '80%'}),
        ], id="frequency-dropdown-container", style={'display': 'none'}),

        html.Div([
            html.Div([
                daq.BooleanSwitch(id="boolean-average", on=False, label="Grouped wells", labelPosition="top"),
                daq.BooleanSwitch(id="boolean-std", on=False, label="Display std", labelPosition="top"),
                html.Div([
                    dcc.Slider(id='std-slider', min=None, max=None, step=None),
                ], id="std-slider-container", style={'display': 'none'}),
            ]),
            dcc.RadioItems(id="normalize-option", options=['AbsZ', 'AbsZ_t0', 'AbsZ_max', 'AbsZ_min_max'], value='AbsZ'),
                dcc.Graph(id='impedence-plot'),
        ], id="impedence-plot-container", style={'display': 'none'}),
    ],
    style={"max-width": "90"},
)

def AbsZ(absz, **kwargs):
    return absz
    
def AbsZ_t0(absz, absz_t0, **kwargs):
    return absz / absz_t0

def AbsZ_max(absz, absz_max, **kwargs):
    return absz / absz_max
    
def AbsZ_min_max(absz, absz_min, absz_max, **kwargs):
    return (absz - absz_min) / (absz_max - absz_min)

def filter_frequency(df, frequency):
    if isinstance(frequency, list):
        return df[df["Frequency"].isin(frequency)]
    elif isinstance(frequency, float) or isinstance(frequency, int):
        return df[df["Frequency"] == frequency]
    else:
        raise ValueError(f"Type of frequency ({type(frequency)}) in data is incorrect.")

def get_figure(df: pandas.DataFrame,
               normalize: str,
               std: bool = False,
               std_step: int = 1,
               **kwargs):
    if len(df["Frequency"].unique()) == 1:
        fig = go.Figure()
        for i, well in enumerate(df["Well"].unique()):
            well_df = df[df["Well"] == well]
            
            color = px.colors.qualitative.Dark24[i]
        
            if std:
                error_color = plotly.colors.hex_to_rgb(color)
                error_color = error_color + (0.5,)
                error_color = "rgba(" + ",".join(map(str, error_color)) + ")"
                
                error_y = dict(type='data',
                               array=well_df[f"{normalize}_std"].values[::std_step],
                               color=error_color,
                               thickness=0.8,
                               width=3)
            else:
                error_y = None
            
            fig.add_trace(go.Scatter(
                x=well_df["dt"].values[::std_step],
                y=well_df[normalize].values[::std_step],
                mode='lines',
                line=dict(color=color),
                name=well,
                error_y=error_y,))
    elif len(df["Frequency"].unique()) > 1:
        # select rows with frequency value
        data = []
        for frq in df["Frequency"].unique():
            dff = df[df['Frequency'] == frq]
            for well in dff["Well"].unique():
                well_df = dff[dff["Well"] == well]
                data.append(go.Scatter3d(x=well_df["dt"].values,
                                         y=np.full(len(well_df["dt"].values), frq),
                                         z=well_df[normalize].values,
                                         name="{} ({:.2f})".format(well, frq),
                                         legendgroup=str(frq),
                                         legendgrouptitle_text="{:.2f} Hz".format(frq),
                                         marker=dict(size=1)))

        fig = go.Figure(data=data)
        fig.update_layout(legend=dict(groupclick="toggleitem"))
    else:
        raise ValueError("Number of 'frequency' in data is wrong.")

    fig.update_layout(autosize=False, width=1200, height=600)
    return fig


@app.callback(
    [Output("frequency-dropdown", "options"), Output("frequency-dropdown", "value"), Output("path", "data")],
    [Input("confirmation-button", "n_clicks"), State('input-path', 'value')],
    background=True,
    prevent_initial_call=True,
)
def update_output(nclicks, path_):
    if os.path.exists(path_):
        experiment_path = path_
    else:
        raise FileNotFoundError("Experiment folder not found ({}).".format(path_))

    base, name = os.path.split(path_)
    save_path = os.path.join(SAVE_DIRECTORY, f"{name}.csv")
    
    if os.path.exists(save_path):
        df = pandas.read_csv(save_path)
    else:
        # 4. Load and Process Data for All Wells
        # This section processes each well in the selected experiment and compiles the data into a combined DataFrame.
        
        # Initialize a list to store DataFrames for each well
        all_wells_data = []
        
        # Automatically detect all wells in the experiment folder
        detected_wells = [name for name in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, name))]
        
        if not detected_wells:
            raise ValueError("No well folders found in experiment folder.")
        
        set_props("animated-progress-bar-container", {'style': {'display': 'block'}})
        
        # Iterate through each detected well and load its data
        for i, well in enumerate(detected_wells):
            well_path = os.path.join(experiment_path, well)
            df = utils.load_well_data(well_path, impedance_parameters)
            if df is not None:
                all_wells_data.append(df)
            else:
                print(f"No data found for well '{well}'.")
    
            # update loading bar
            perc = int(((i+1) / len(detected_wells)) * 100)
            set_props("animated-progress-bar", {'value': perc, 'label': "{}%".format(perc),'animated': True, 'striped': True})
    
        # disable progress bar
        set_props("animated-progress-bar-container", {'style': {'display': 'none'}})
        
        # Check if any well data was loaded
        if not all_wells_data:
            raise ValueError("No data loaded for any of the wells in the selected experiment.")
        
        # Combine all wells' DataFrames into a single DataFrame
        df = pandas.concat(all_wells_data, ignore_index=True)
    
        # This step ensures that the DataFrame is complete and saves it to a CSV file for later use.
        df = utils.check_data(df)
        df["Date"] = df["Date"].dt.strftime(fmt)
    
        # normalize data
        set_props("process-progress-bar-container", {'style': {'display': 'block'}})
        for j, frq in enumerate(df["Frequency"].unique()[:10]):
            for well in df["Well"].unique():
                dff = df[(df["Frequency"] == frq) & (df["Well"] == well)]
                
                absz_max = dff["AbsZ"].max()
                absz_min = dff["AbsZ"].min()
                t0 = datetime.strptime(dff["Date"].min(), fmt)
                absz_t0 = dff.loc[dff[dff["Date"] == dff["Date"].min()].index[0], "AbsZ"]
                
                for idx, d in enumerate(sorted(dff["Date"])):
                    i = dff[dff["Date"] == d].index.values[0]
                    dt = datetime.strptime(dff.loc[i, "Date"], fmt) - t0
                    
                    df.loc[i, "index"] = idx
                    df.loc[i, "dt"] = (dt.days * 24 * 60) + (dt.seconds / 60)
                    df.loc[i, "AbsZ_t0"] = AbsZ_t0(df.loc[i, "AbsZ"], absz_t0)
                    df.loc[i, "AbsZ_max"] = AbsZ_max(df.loc[i, "AbsZ"], absz_max)
                    df.loc[i, "AbsZ_min_max"] = AbsZ_min_max(df.loc[i, "AbsZ"], absz_min, absz_max)
    
            # update loading bar
            perc = int(((j+1) / len(df["Frequency"].unique())) * 100)
            set_props("process-progress-bar", {'value': perc, 'label': "{}%".format(perc),'animated': True, 'striped': True})
    
        # hide progress bars
        set_props("process-progress-bar-container", {'style': {'display': 'none'}})
    
        # save to csv
        df.to_csv(save_path)

    set_props("frequency-dropdown-container", {'style': {'display': 'block'}})
    frqs = df["Frequency"].unique()
    return frqs, frqs[0], save_path


@app.callback(
    [Output("std-slider-container", "style"), Output("std-slider", "min"), Output("std-slider", "max"), 
     Output("std-slider", "step"), Output("std-slider", "value")],
    Input("boolean-std", "on"),
    prevent_initial_call=True,
)
def update_std_slider(value):
    if value:
        # show slider
        return {'display': 'block'}, 1, 50, 5, 1
    else:
        # hide slider
        return {'display': 'none'}, None, None, None, None


@app.callback(
    [Output("impedence-plot", "figure"), Output("impedence-plot-container", "style"), Output("groups", "data")],
    [
        Input("boolean-average", "on"),
        Input("boolean-std", "on"),
        Input("std-slider", "value"),
        Input("frequency-dropdown", "value"),
        Input("normalize-option", "value"),
        State('path', 'data'),
        State("groups", "data"),
    ],
    #background=True,   # this may cause PID problems
    prevent_initial_call=True,
)
def update_plot(average_boolean, std_boolean, std_value, frequency, normalize, path_, groups):
    # show spinner
    #set_props("spinner-container", {'style': {'display': 'block'}})

    # load data
    df = pandas.read_csv(path_)

    # filter frequencies
    df = filter_frequency(df, frequency)
        
    if average_boolean:
        if groups is None:
            # create groups of wells
            groups = df["Well"].unique()
            groups = np.array_split(groups, indices_or_sections=math.ceil(len(groups) / GROUPS_SIZE))

        # average values of impedence
        groups_df = pandas.DataFrame()
        for group in groups:
            group_name = "(" + ",".join(group) + ")"
            for frq in df["Frequency"].unique():
                dfff = df[(df["Well"].isin(group)) & (df["Frequency"] == frq)]
                for idx in dfff["index"].unique():
                    group_idx_df = dfff[dfff["index"] == idx]
                    new_line = dict(group_idx_df.iloc[0])
                    new_line["Well"] = group_name
                    new_line[normalize] = np.mean(group_idx_df[normalize])
                    new_line.update({f"{normalize}_std" : np.std(group_idx_df[normalize])})
                    groups_df = pandas.concat((groups_df, pandas.DataFrame([new_line])), ignore_index=True)
        
        # replace DataFrame with groups DataFrame
        df = groups_df
        
        # create figure with std bars
        fig = get_figure(df=df, normalize=normalize, std=std_boolean, std_step=std_value)
    else:
        fig = get_figure(df=df, normalize=normalize)
        
    # hide spinner
    #set_props("spinner-container", {'style': {'display': 'none'}})
    
    return fig, {'display': 'block'}, groups


if __name__ == "__main__":
    url = "127.0.0.1"
    port = 8050
    webbrowser.open_new(f'http://{url}:{port}/')
    # Set debug=False before compiling into exe file
    app.run_server(debug=True, host=url, port=port)
