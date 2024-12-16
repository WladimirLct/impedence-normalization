import os
import pandas
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import dash
from dash import set_props, DiskcacheManager, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

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
        dcc.Store(id='data'),
        html.H1("File Browser"),
        html.H2("Set path"),
        html.Div([
            dcc.Input(id="input-path", type="text", debounce=True, style={'width' : '70%'}),
            dbc.Button("Confirmation", id="confirmation-button", n_clicks=0),
        ]),
        
        html.Div([
            dbc.Progress(id="animated-progress-bar", style={'width' : '70%'}),
        ], id="animated-progress-bar-container", style={'display': 'none'}),
        
        html.Div([
            dcc.Dropdown([], [], id="frequency-dropdown", multi=True),
        ], id="frequency-dropdown-container", style={'display': 'none'}),

        html.Div([
            dbc.Progress(id="process-progress-bar", style={'width' : '70%'}),
        ], id="process-progress-bar-container", style={'display': 'none'}),

        html.Div([
            dcc.RadioItems(id="normalize-option", options=['raw', 't0', 'impedence-max'], value='raw'),
            dcc.Graph(id='impedence-plot'),
        ], id="impedence-plot-container", style={'display': 'none'}),
    ],
    style={"max-width": "90"},
)


@app.callback(
    [Output("frequency-dropdown", "options"), Output("frequency-dropdown", "value"), Output("data", "data")],
    [Input("confirmation-button", "n_clicks"), State('input-path', 'value')],
    background=True,
    prevent_initial_call=True,
)
def update_output(nclicks, path_):
    if os.path.exists(path_):
        experiment_path = path_
    else:
        raise FileNotFoundError("Experiment folder not found ({}).".format(path_))

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
    for i, well in enumerate(detected_wells):                    ####   !!!!!!    WARNING: remove [:2]  !!!!!!   #####
        well_path = os.path.join(experiment_path, well)
        df = utils.load_well_data(well_path, impedance_parameters)
        if df is not None:
            all_wells_data.append(df)
        else:
            print(f"No data found for well '{well}'.")

        # update loading bar
        perc = int(((i+1) / len(detected_wells)) * 100)
        set_props("animated-progress-bar", {'value': perc, 'label': "{}%".format(perc),'animated': True, 'striped': True})
    
    set_props("animated-progress-bar-container", {'style': {'display': 'none'}})
    
    # Check if any well data was loaded
    if not all_wells_data:
        raise ValueError("No data loaded for any of the wells in the selected experiment.")
    
    # Combine all wells' DataFrames into a single DataFrame
    df = pandas.concat(all_wells_data, ignore_index=True)

    # This step ensures that the DataFrame is complete and saves it to a CSV file for later use.
    df = utils.check_data(df)
    df["Date"] = df["Date"].dt.strftime(fmt)
    
    # save to csv
    base, name = os.path.split(path_)
    df.to_csv(os.path.join(SAVE_DIRECTORY, f"{name}.csv"))

    set_props("frequency-dropdown-container", {'style': {'display': 'block'}})

    frqs = df["Frequency"].unique()
    return frqs, frqs[0], df.to_dict("records")


@app.callback(
    Output("impedence-plot", "figure"),
    [Input("frequency-dropdown", "value"), Input("normalize-option", "value"), State('data', 'data')],
    background=True,
    prevent_initial_call=True,
)
def update_plot(value, normalize, data):
    df = pandas.DataFrame(data)
    
    set_props("process-progress-bar-container", {'style': {'display': 'block'}})
    
    if isinstance(value, int):
        dff = df[df["Frequency"] == value]
        for j, well in enumerate(dff["Well"].unique()):
            dfff = dff[dff["Well"] == well]
        
            absz_max = dfff["AbsZ"].max()

            t0 = datetime.strptime(dfff["Date"].min(), fmt)
            absz_t0 = dfff.loc[dfff[dfff["Date"] == dfff["Date"].min()].index[0], "AbsZ"]
        
            for idx, d in enumerate(sorted(dfff["Date"])):
                i = dfff[dfff["Date"] == d].index.values[0]
                dt = datetime.strptime(dfff.loc[i, "Date"], fmt) - t0
                dff.loc[i, "dt"] = (dt.days * 24 * 60) + (dt.seconds / 60)
                dff.loc[i, "index"] = idx
                dff.loc[i, "normt0AbsZ"] = dff.loc[i, "AbsZ"] / absz_t0
                dff.loc[i, "normmaxAbsZ"] = dff.loc[i, "AbsZ"] / absz_max

            # update loading bar
            perc = int(((j+1) / len(dff["Well"].unique())) * 100)
            set_props("process-progress-bar", {'value': perc, 'label': "{}%".format(perc),'animated': True, 'striped': True})

        if normalize == "t0":
            fig = px.line(dff, x="dt", y="normt0AbsZ", color="Well")
        elif normalize == "impedence-max":
            fig = px.line(dff, x="dt", y="normmaxAbsZ", color="Well")
        else:
            fig = px.line(dff, x="dt", y="AbsZ", color="Well")
    elif isinstance(value, list):
        # select row with frequency value
        dff = df[df['Frequency'].isin(value)]
        data = []
        for j, frq in enumerate(dff["Frequency"].unique()):
            for well in dff["Well"].unique():
                dfff = dff[(dff["Frequency"] == frq) & (dff["Well"] == well)]
                
                absz_max = dfff["AbsZ"].max()
                t0 = datetime.strptime(dfff["Date"].min(), fmt)
                absz_t0 = dfff.loc[dfff[dfff["Date"] == dfff["Date"].min()].index[0], "AbsZ"]
                
                for idx, d in enumerate(sorted(dfff["Date"])):
                    i = dfff[dfff["Date"] == d].index.values[0]
                    dt = datetime.strptime(dfff.loc[i, "Date"], fmt) - t0
                    
                    dfff.loc[i, "dt"] = (dt.days * 24 * 60) + (dt.seconds / 60)
                    dfff.loc[i, "index"] = idx
                    if normalize == "t0":
                        dfff.loc[i, "AbsZ"] = dfff.loc[i, "AbsZ"] / absz_t0
                    elif normalize == "impedence-max":
                        dfff.loc[i, "AbsZ"] = dfff.loc[i, "AbsZ"] / absz_max                        
                data.append(go.Scatter3d(x=dfff["dt"].values,
                                         y=np.full(len(dfff["dt"].values), frq),
                                         z=dfff["AbsZ"].values,
                                         name=well,
                                         marker=dict(size=1)))

            # update loading bar
            perc = int(((j+1) / len(dff["Frequency"].unique())) * 100)
            set_props("process-progress-bar", {'value': perc, 'label': "{}%".format(perc),'animated': True, 'striped': True})
        
        fig = fig = go.Figure(data=data)
    else:
        raise ValueError("Variable 'value' given is neither scalar or list.")

    # disable loading bar
    set_props("process-progress-bar", {'style': {'display': 'none'}})

    # show plot figure container
    set_props("impedence-plot-container", {'style': {'display': 'block'}})
    
    return  fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)