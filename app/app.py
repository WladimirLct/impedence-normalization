import os
import pandas
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# format for date hours
fmt = '%Y-%m-%d %H:%M:%S'

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], background_callback_manager=background_callback_manager)
app.layout = html.Div(
    [
        dcc.Store(id='path'),
        html.H1("File Browser"),
        html.H2("Set path"),
        html.Div([
            dcc.Input(id="input-path", type="text", debounce=True, style={'width' : '70%'}),
            dbc.Button("Confirmation", id="confirmation-button", n_clicks=0),
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
            dcc.Dropdown([], [], id="frequency-dropdown", multi=True),
        ], id="frequency-dropdown-container", style={'display': 'none'}),

        html.Div([
            html.Div([
                daq.BooleanSwitch(id="boolean-average", on=True, label="Grouped wells", labelPosition="top"),
                daq.BooleanSwitch(id="boolean-std", on=True, label="Display std", labelPosition="top"),
            ]),
            dcc.RadioItems(id="normalize-option", options=['raw', 't0', 'impedence-max'], value='raw'),
            dcc.Graph(id='impedence-plot'),
        ], id="impedence-plot-container", style={'display': 'none'}),
    ],
    style={"max-width": "90"},
)


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
                t0 = datetime.strptime(dff["Date"].min(), fmt)
                absz_t0 = dff.loc[dff[dff["Date"] == dff["Date"].min()].index[0], "AbsZ"]
                
                for idx, d in enumerate(sorted(dff["Date"])):
                    i = dff[dff["Date"] == d].index.values[0]
                    dt = datetime.strptime(dff.loc[i, "Date"], fmt) - t0
                    
                    df.loc[i, "dt"] = (dt.days * 24 * 60) + (dt.seconds / 60)
                    df.loc[i, "AbsZ_t0"] = df.loc[i, "AbsZ"] / absz_t0
                    df.loc[i, "AbsZ_max"] = df.loc[i, "AbsZ"] / absz_max
    
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
    Output("impedence-plot", "figure"),
    [Input("frequency-dropdown", "value"), Input("normalize-option", "value"), State('path', 'data')],
    background=True,
    prevent_initial_call=True,
)
def update_plot(value, normalize, path_):
    df = pandas.read_csv(path_)
    
    if isinstance(value, int):
        dff = df[df["Frequency"] == value]

        if normalize == "t0":
            fig = px.line(dff, x="dt", y="AbsZ_t0", color="Well")
        elif normalize == "impedence-max":
            fig = px.line(dff, x="dt", y="AbsZ_max", color="Well")
        else:
            fig = px.line(dff, x="dt", y="AbsZ", color="Well")
    elif isinstance(value, list):
        if normalize == "t0":
            z = "AbsZ_t0"
        elif normalize == "impedence-max":
            z = "AbsZ_max"
        else:
            z = "AbsZ"
        
        # select rows with frequency value
        data = []
        for frq in value:
            dff = df[df['Frequency'] == frq]
            for well in dff["Well"].unique():
                dfff = dff[dff["Well"] == well]
                data.append(go.Scatter3d(x=dfff["dt"].values,
                                         y=np.full(len(dfff["dt"].values), frq),
                                         z=dfff[z].values,
                                         name="{} ({:.2f})".format(well, frq),
                                         legendgroup=str(frq),
                                         legendgrouptitle_text="{:.2f} Hz".format(frq),
                                         marker=dict(size=1)))

        fig = go.Figure(data=data)
        fig.update_layout(autosize=False, width=1200, height=800)
        fig.update_layout(legend=dict(groupclick="toggleitem"))
    else:
        raise ValueError("Variable 'value' given is neither scalar or list.")

    # show plot figure container
    set_props("impedence-plot-container", {'style': {'display': 'block'}})
    
    return  fig

# TODO
# @app.callback(
#     Output("impedence-plot", "figure"),
#     [Input("boolean-average", "on"), State("normalize-option", "value"), State('path', 'data')],
#     prevent_initial_call=True,
# )
# def update_plot_average(value, normalize, path_):
#     pass

# TODO
# @app.callback(
#     Output("impedence-plot", "figure"),
#     [Input("boolean-std", "on"), State("normalize-option", "value"), State('path', 'data')],
#     prevent_initial_call=True,
# )
# def update_plot_std(value, normalize, path_):
#     pass

if __name__ == "__main__":
    # Set debug=False before compiling into exe file
    app.run_server(debug=True, port=8050)