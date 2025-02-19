# stat_analysis.py

import dash
from dash import Input, Output, State, dcc, html, callback  # Added 'dcc' and 'callback' imports
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np

from io import StringIO
from scipy.stats import ttest_ind, f_oneway

import statsmodels.api as sm

from app_instance import app, data_cache  # Import both app and cache from app_instance

import plotly.express as px  # Import Plotly Express for plotting

# Stats page layout
stats_layout = dbc.Container([

    dcc.Store(id='session-store', storage_type='session'),
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Statistical Analysis", className="h5"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Well 1:", className="mb-2"),
                            dcc.Dropdown(
                                id="stats-well1-selector",
                                placeholder="Select first well...",
                                className="mb-3"
                            ),
                            dbc.Label("Select Frequency for Well 1:", className="mb-2"),
                            dcc.Dropdown(
                                id="stats-well1-freq-selector",
                                placeholder="Select frequency...",
                                className="mb-3"
                            ),
                        ], md=4),

                        dbc.Col([
                            dbc.Label("Select Well 2:", className="mb-2"),
                            dcc.Dropdown(
                                id="stats-well2-selector",
                                placeholder="Select second well...",
                                className="mb-3"
                            ),
                            dbc.Label("Select Frequency for Well 2:", className="mb-2"),
                            dcc.Dropdown(
                                id="stats-well2-freq-selector",
                                placeholder="Select frequency...",
                                className="mb-3"
                            ),
                        ], md=4),

                        dbc.Col([
                            dbc.Label("Test Type:", className="mb-2"),
                            dcc.Dropdown(
                                id="stats-test-selector",
                                options=[
                                    {'label': 'ANOVA', 'value': 'anova'},
                                    {'label': "Student's t-test", 'value': 'ttest'}
                                ],
                                value='ttest',
                                clearable=False,
                                className="mb-3"
                            ),
                        ], md=4)
                    ]),

                    dbc.Button(
                        "Run Statistical Analysis",
                        id="run-stats-button",
                        color="primary",
                        className="w-100 mb-3"
                    ),

                    dbc.Spinner(html.Div(id="stats-results-output")),

                    html.Hr(),

                    dbc.Button(
                        "Download Full Report",
                        id="download-report-button",
                        color="success",
                        className="w-100",
                        disabled=True
                    ),
                    dcc.Download(id="download-report"),

                    dbc.Button(
                        "Return to Main Page",
                        id="return-button",
                        color="secondary",
                        className="w-100 mt-3",
                        outline=True
                    ),
                ])
            ]),
            md=12  # Adjusted to full width for better layout
        )
    ], 
    
    className="g-4"),
    

], fluid=True)

# Define callbacks specific to the stats page
def register_stats_callbacks(app):
    # Navigation callback to return to main page
    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        Input('return-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def return_to_main(n_clicks):
        if n_clicks:
            return '/'
        else:
            raise dash.exceptions.PreventUpdate

    # Callback to update well selectors
    @app.callback(
        [Output("stats-well1-selector", "options"),
         Output("stats-well2-selector", "options")],
        Input("session-store", "data")
    )
    def update_stats_wells(session_id):
        if not session_id:
            return [], []
        df = data_cache.get(session_id)
        if df is None:
            return [], []
        wells = sorted(df['Well'].unique())
        options = [{'label': w, 'value': w} for w in wells]
        return options, options

    # Callbacks to update frequency selectors based on selected wells
    @app.callback(
        [Output("stats-well1-freq-selector", "options"),
         Output("stats-well1-freq-selector", "value")],
        Input("stats-well1-selector", "value"),
        State("session-store", "data")
    )
    def update_well1_freq_selector(well1, session_id):
        if not well1 or not session_id:
            return [], None
        df = data_cache.get(session_id)
        if df is None:
            return [], None
        frequencies = sorted(df[df['Well'] == well1]['Frequency'].unique())
        options = [{'label': 'All', 'value': 'All'}] + [{'label': f"{f} Hz", 'value': f} for f in frequencies]
        return options, 'All'

    @app.callback(
        [Output("stats-well2-freq-selector", "options"),
         Output("stats-well2-freq-selector", "value")],
        Input("stats-well2-selector", "value"),
        State("session-store", "data")
    )
    def update_well2_freq_selector(well2, session_id):
        if not well2 or not session_id:
            return [], None
        df = data_cache.get(session_id)
        if df is None:
            return [], None
        frequencies = sorted(df[df['Well'] == well2]['Frequency'].unique())
        options = [{'label': 'All', 'value': 'All'}] + [{'label': f"{f} Hz", 'value': f} for f in frequencies]
        return options, 'All'

    # Modify run_statistical_analysis to use selected frequencies
    @app.callback(
        [Output("stats-results-output", "children"),
        Output("download-report-button", "disabled")],
        Input("run-stats-button", "n_clicks"),
        [State("session-store", "data"),
        State("stats-well1-selector", "value"),
        State("stats-well2-selector", "value"),
        State("stats-well1-freq-selector", "value"),
        State("stats-well2-freq-selector", "value"),
        State("stats-test-selector", "value")],
        prevent_initial_call=True
    )
    def run_statistical_analysis(n_clicks, session_id, well1, well2, freq1, freq2, test_type):
        if not all([session_id, well1, well2, freq1, freq2]):
            return dash.no_update, True

        df = data_cache.get(session_id)
        if df is None:
            return dash.no_update, True

        # Filter data for each well and frequency
        well1_data = df[df['Well'] == well1]
        if freq1 != 'All':
            well1_data = well1_data[well1_data['Frequency'] == freq1]

        well2_data = df[df['Well'] == well2]
        if freq2 != 'All':
            well2_data = well2_data[well2_data['Frequency'] == freq2]

        try:
            if test_type == 'ttest':
                result = perform_t_test(well1_data, well2_data)

                if 'result_text' in result:
                    # Display numerical results when few frequencies
                    results_output = html.Div([
                        html.H5("Student's T-test Results"),
                        html.Pre(result['result_text'], className="text-wrap")
                    ])
                else:
                    # Display plots when multiple frequencies
                    results_output = html.Div([
                        html.H5("Student's T-test Results"),
                        dcc.Graph(figure=result['t_stat_fig']),
                        dcc.Graph(figure=result['p_value_fig'])
                    ])

                # Generate report text
                report_text = generate_report_text_ttest(result['summary_df'], well1, well2)

            elif test_type == 'anova':
                result = perform_anova(well1_data, well2_data)

                if 'result_text' in result:
                    # Display numerical results when few frequencies
                    results_output = html.Div([
                        html.H5("ANOVA Results"),
                        html.Pre(result['result_text'], className="text-wrap")
                    ])
                else:
                    # Display plots when multiple frequencies
                    results_output = html.Div([
                        html.H5("ANOVA Results"),
                        dcc.Graph(figure=result['f_stat_fig']),
                        dcc.Graph(figure=result['p_value_fig'])
                    ])

                # Generate report text
                report_text = generate_report_text_anova(result['summary_df'], well1, well2)

            # Since you've removed Poisson and linear regression, no need to handle them

            # Remove all reports that arent from this session_id
            for key in data_cache.keys():
                if key.endswith("_report") and key != f"{session_id}_report":
                    del data_cache[key]
            # Store the report text in cache for the download
            data_cache[f"{session_id}_report"] = report_text

            return results_output, False  # Enable download button

        except Exception as e:
            error_message = f"Error in analysis: {str(e)}"
            return dbc.Alert(error_message, color="danger"), True

    # Callback to generate the report
    @app.callback(
        Output("download-report", "data"),
        Input("download-report-button", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def generate_report(n_clicks, session_id):
        if n_clicks:
            report_text = data_cache.get(f"{session_id}_report")
            return dcc.send_string(report_text, "analysis_report.txt")
        else:
            raise dash.exceptions.PreventUpdate

# Statistical analysis functions
def perform_t_test(w1, w2):
    common_freq = np.intersect1d(w1['Frequency'].unique(), w2['Frequency'].unique())
    if len(common_freq) == 0:
        raise ValueError("No matching frequencies to perform the t-test.")

    t_stats = []
    p_values = []
    freqs = []

    for freq in common_freq:
        w1_freq = w1[w1['Frequency'] == freq]['AbsZ']
        w2_freq = w2[w2['Frequency'] == freq]['AbsZ']
        t_stat, p_val = ttest_ind(w1_freq, w2_freq, equal_var=False)
        t_stats.append(t_stat)
        p_values.append(p_val)
        freqs.append(freq)

    # Prepare DataFrame for reporting
    summary_df = pd.DataFrame({
        'Frequency': freqs,
        'T-statistic': t_stats,
        'P-value': p_values
    })

    if len(freqs) <= 2:
        # Return numerical results for few frequencies
        result_text = "Student's T-test Results:\n"
        for _, row in summary_df.iterrows():
            result_text += (f"Frequency {row['Frequency']:.2f} Hz: "
                            f"T-statistic = {row['T-statistic']:.3f}, "
                            f"P-value = {row['P-value']:.4f}\n")
        return {
            'summary_df': summary_df,
            'result_text': result_text
        }
    else:
        # Generate plots for multiple frequencies
        summary_df.sort_values('Frequency', inplace=True)

        t_stat_fig = px.line(summary_df, x='Frequency', y='T-statistic',
                             title="T-statistic across Frequencies",
                             markers=True,
                             labels={'Frequency': 'Frequency (Hz)'})

        p_value_fig = px.line(summary_df, x='Frequency', y='P-value',
                              title="P-values across Frequencies",
                              markers=True,
                              labels={'Frequency': 'Frequency (Hz)'})

        # Add reference line at p=0.05
        p_value_fig.add_hline(y=0.05, line_dash="dash",
                              annotation_text="Significance level (p=0.05)",
                              annotation_position="bottom right")

        return {
            'summary_df': summary_df,
            't_stat_fig': t_stat_fig,
            'p_value_fig': p_value_fig
        }

def perform_anova(w1, w2):
    common_freq = np.intersect1d(w1['Frequency'].unique(), w2['Frequency'].unique())
    if len(common_freq) == 0:
        raise ValueError("No matching frequencies to perform ANOVA.")

    f_stats = []
    p_values = []
    freqs = []

    for freq in common_freq:
        groups = [
            w1[w1['Frequency'] == freq]['AbsZ'],
            w2[w2['Frequency'] == freq]['AbsZ']
        ]
        f_stat, p_val = f_oneway(*groups)
        f_stats.append(f_stat)
        p_values.append(p_val)
        freqs.append(freq)

    # Prepare DataFrame for reporting
    summary_df = pd.DataFrame({
        'Frequency': freqs,
        'F-statistic': f_stats,
        'P-value': p_values
    })

    if len(freqs) <= 2:
        # Return numerical results for few frequencies
        result_text = "ANOVA Results:\n"
        for _, row in summary_df.iterrows():
            result_text += (f"Frequency {row['Frequency']:.2f} Hz: "
                            f"F-statistic = {row['F-statistic']:.3f}, "
                            f"P-value = {row['P-value']:.4f}\n")
        return {
            'summary_df': summary_df,
            'result_text': result_text
        }
    else:
        # Generate plots for multiple frequencies
        summary_df.sort_values('Frequency', inplace=True)

        f_stat_fig = px.line(summary_df, x='Frequency', y='F-statistic',
                             title="F-statistic across Frequencies",
                             markers=True,
                             labels={'Frequency': 'Frequency (Hz)'})

        p_value_fig = px.line(summary_df, x='Frequency', y='P-value',
                              title="P-values across Frequencies",
                              markers=True,
                              labels={'Frequency': 'Frequency (Hz)'})

        # Add reference line at p=0.05
        p_value_fig.add_hline(y=0.05, line_dash="dash",
                              annotation_text="Significance level (p=0.05)",
                              annotation_position="bottom right")

        return {
            'summary_df': summary_df,
            'f_stat_fig': f_stat_fig,
            'p_value_fig': p_value_fig
        }
    
# Functions to generate reports
def generate_report_text_ttest(summary_df, well1, well2):
    report_lines = [
        f"Student's T-test Report",
        f"Comparing {well1} and {well2}",
        "",
        "Results:"
    ]
    for _, row in summary_df.iterrows():
        report_lines.append(
            f"Frequency {row['Frequency']:.2f} Hz: T-statistic = {row['T-statistic']:.3f}, P-value = {row['P-value']:.4f}"
        )
    interpretation = "\nInterpretation:\nP-values less than 0.05 indicate a statistically significant difference."
    report_text = '\n'.join(report_lines) + interpretation
    return report_text

def generate_report_text_anova(summary_df, well1, well2):
    report_lines = [
        f"ANOVA Report",
        f"Comparing {well1} and {well2}",
        "",
        "Results:"
    ]
    for _, row in summary_df.iterrows():
        report_lines.append(
            f"Frequency {row['Frequency']:.2f} Hz: F-statistic = {row['F-statistic']:.3f}, P-value = {row['P-value']:.4f}"
        )
    interpretation = "\nInterpretation:\nP-values less than 0.05 suggest significant differences between groups."
    report_text = '\n'.join(report_lines) + interpretation
    return report_text