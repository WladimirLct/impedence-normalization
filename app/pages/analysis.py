# File: pages/analysis.py

import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

from callbacks.analysis_page import *

dash.register_page(__name__)

layout = html.Div([
    dcc.Location(id='url-p3', refresh='callback-nav'),

    # Stores pour les données ou paramètres
    dcc.Store(id='groups_enabled', storage_type='memory'),

    dcc.Store(id='list_well_not_in_group', storage_type='memory'),
    dcc.Store(id='single_well_groups_data', storage_type='memory'),
    dcc.Store(id='well_groups_data', storage_type='memory'),
    dcc.Store(id='wells', storage_type='memory'),
 
    dcc.Store(id='selected_normalization', storage_type='memory', data='absz'),
    dcc.Store(id='selected_frequency', storage_type='memory'),
    dcc.Store(id='std_bar_frequency', storage_type='memory', data=10),

    html.Div([
        # Information header
        html.Div([
            html.Div([
                html.Pre([
                    dcc.Link("Accueil", href='/', className="fw-normal text-primary", id="info-analysis-1")
                ], className="mb-0 fw-bold")
            ], className="text-center bg-light rounded p-3"),
            html.Div([
                html.Pre([
                    "Expérience: ",
                    html.Span("", className="fw-normal text-primary", id="info-analysis-1")
                ], className="mb-0 fw-bold")
            ], className="text-center bg-light rounded p-3"),
            html.Div([
                html.Pre([
                    "Puits: ",
                    html.Span("", className="fw-normal text-primary", id="info-analysis-2")
                ], className="mb-0 fw-bold")
            ], className="text-center bg-light rounded p-3"),
            html.Div([
                html.Pre([
                    "Fichiers: ",
                    html.Span("", className="fw-normal text-primary", id="info-analysis-3")
                ], className="mb-0 fw-bold")
            ], className="text-center bg-light rounded p-3"),
            html.Div([
                html.Pre([
                    "Début : ",
                    html.Span("", className="fw-normal text-primary", id="info-analysis-4")
                ], className="mb-0 fw-bold")
            ], className="text-center bg-light rounded p-3"),
            html.Div([
                html.Pre([
                    "Durée (h): ",
                    html.Span("", className="fw-normal text-primary", id="info-analysis-5")
                ], className="mb-0 fw-bold")
            ], className="text-center bg-light rounded p-3"),
        ], className="d-flex justify-content-between gap-3 mb-4"),

        html.Div([
            html.Div([
                # Download button section
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-download me-2"), "Télécharger figures & infos"], 
                        id='download-figures-info', 
                        color="warning",
                        className='w-100'
                    ),
                    dcc.Download(id='download-component')
                ], className='p-4 bg-white rounded shadow-sm border mb-4'),
                
                # Add group section
                html.Div([
                    html.H5("Ajouter un groupe", className="mb-4 text-dark fw-bold border-bottom pb-2"),
                    
                    dcc.Input(
                        id='input_name_group', 
                        type='text', 
                        placeholder='Nom du groupe', 
                        className="form-control mb-3"
                    ),

                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Sélection des puits", className="form-label small text-muted mb-2"),
                                dcc.Dropdown(
                                    id='select_wells', 
                                    options=[], 
                                    multi=True, 
                                    value=[], 
                                    placeholder='Choisir les puits',
                                    className="mb-3"
                                )
                            ]),
                            dbc.Col([
                                dbc.Label("Type de groupe", className="form-label small text-muted mb-2"),
                                dcc.Dropdown(
                                    id='label_group', 
                                    options=[
                                        {'label': 'Infecté traité', 'value': 'inf-trt'},
                                        {'label': 'Infecté non-traité', 'value': 'inf-ntrt'},
                                        {'label': 'Contrôle', 'value': 'controle'},
                                    ], 
                                    placeholder='Label',
                                    className="mb-3"
                                )
                            ])
                        ], className="gx-3"),

                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Concentration", className="form-label small text-muted mb-2"),
                                dcc.Input(
                                    id='input_concentration', 
                                    type='number', 
                                    placeholder='Concentration', 
                                    className="form-control",
                                    value=0,
                                )
                            ]),
                            dbc.Col([
                                dbc.Label("Unité", className="form-label small text-muted mb-2"),
                                dcc.Dropdown(
                                    id='select_unit', 
                                    options=[{'label': u, 'value': u} for u in ['mM', 'µM', 'nM']], 
                                    placeholder='Unité',
                                    value='mM',
                                )
                            ])
                        ], className="gx-3"),
                    ]),
                    
                    dbc.Button(
                        [html.I(className="fas fa-plus me-2"), "Ajouter le groupe"], 
                        id='add-groups', 
                        color="success",
                        className='mt-4 w-100'
                    )
                ], className='p-4 bg-white rounded shadow-sm border mb-4'),

                # Groups display section
                html.Div([
                    html.H5("Groupes configurés", className="mb-4 text-dark fw-bold border-bottom pb-2"),
                    html.Div(id='well-groups', className='p-2')
                ], className="p-4 bg-white rounded shadow-sm border"),
            ], className="col-lg-3 col-md-12"),


            # Main content area
            html.Div([
                # Graph section
                html.Div([
                    html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Normalisation", className="form-label fw-bold mb-3"),
                                        dcc.Dropdown(id='select_norm', options=[], value="absz", placeholder='Choisir une normalisation', className="mb-3"),
                                    ], width=3),

                                    dbc.Col([
                                        dbc.Label("Fréquence", className="form-label fw-bold mb-3"),
                                        dcc.Dropdown(id='select_frequency', options=[], value="", placeholder='Choisir une fréquence', className="mb-3"),
                                    ], width=3),

                                    dbc.Col([
                                        dbc.Label("Fréquence barres σ", className="form-label fw-bold mb-3"),
                                        dbc.InputGroup([
                                            dbc.Input(id='std-bar-frequency', type='number', value=10, min=0, placeholder="-", className="form-control"
                                            ),
                                            dbc.InputGroupText("points")
                                        ], className="mb-3"),
                                    ], width=2),

                                    dbc.Col([
                                        dbc.Label("T0", className="form-label fw-bold mb-3"),
                                        dbc.InputGroup([
                                                dbc.Input(id='select_t0', type='number', value=0, min=0, step=0.01, placeholder="-", className="form-control"
                                            ),
                                            dbc.InputGroupText("heures")
                                        ], className="mb-3"),
                                    ], width=2),

                                    dbc.Col([
                                        dbc.Button(
                                            [html.I(className="fas fa-play me-2"), "Appliquer"], 
                                            id='apply-filters', 
                                            color="primary",
                                            className="mb-3 w-100"
                                        ),
                                    ], width=2),
                                ], align='end')
                            ])
                        ], className="mb-4"),

                        dbc.Button([
                                html.I(className="fas fa-chart-line me-2"),
                                html.Span("Graphique", className="fw-bold"),
                                html.I(id='collapse_graph_state', className="fas fa-chevron-up ms-auto")
                            ], id="collapse_graph_btn", color="light", className="w-100 d-flex align-items-center justify-content-between text-dark border", n_clicks=1
                        ),

                        dbc.Collapse([
                            html.Div([
                                dcc.Graph(
                                    id='main-graph', 
                                    config={'displayModeBar': True}, 
                                    className="border rounded",
                                    style={'height': '550px'}
                                ),
                            ])
                        ], id="collapse_graph", className="mt-3"),

                        ## Statistics section
                        dbc.Button([
                                html.I(className="fas fa-chart-bar me-2"),
                                html.Span("Analyses", className="fw-bold"), # ID added
                                html.I(id='collapse_stats_state', className="fas fa-chevron-down ms-auto")
                            ],
                            id="collapse_stats_btn", color="light", className="w-100 d-flex align-items-center justify-content-between text-dark border mt-4", n_clicks=0
                        ),

                        dbc.Collapse([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        # Group selection
                                        dbc.Col([
                                            dbc.Label("Groupe de référence", className="form-label fw-bold"),
                                            dcc.Dropdown(id='select-group-ref', placeholder='Choisir un groupe')
                                        ], width=2),

                                        dbc.Col([
                                            dbc.Label("Groupes de comparaison", className="form-label fw-bold"),
                                            dcc.Dropdown(id='select-group-comp', multi=True, placeholder='Choisir un ou plusieurs groupes')
                                        ], width=4),
                                        
                                        dbc.Col([
                                            dbc.Label("Test statistique", className="form-label fw-bold"),
                                            dcc.Dropdown(
                                                id='select_test',
                                                options=[
                                                    {'label': 'Student t-test',    'value': 'ttest'},
                                                    {'label': 'Mann-Whitney U',    'value': 'mannwhitney'},
                                                ],
                                                value='ttest',  # default
                                                clearable=False,
                                            )
                                        ], width=2),

                                        # P-value threshold input
                                        dbc.Col([
                                            dbc.Label("Seuil de p-value", className="form-label fw-bold"),
                                            dcc.Input(
                                                id='p_value_threshold',
                                                type='number',
                                                value=0.05,
                                                min=0,
                                                max=1,
                                                step=0.01,
                                                className="form-control"
                                            )
                                        ], width=2),
                                        
                                        # Action button
                                        dbc.Col([
                                            dbc.Label("Actions", className="form-label text-white", style={'visibility': 'hidden'}), # Invisible label for alignment
                                            dbc.Button(
                                                [html.I(className="fas fa-play me-2"), "Analyser"], 
                                                id='apply-filters_stats', 
                                                color="primary",
                                                className='w-100'
                                            ),
                                        ], width=2),
                                    ], align="end")
                                ])
                            ], className="mb-4"),

                            # This is where the statistical results will be displayed
                            html.Div(children=[
                                dcc.Graph(id='stats-graph', style={'display': 'none'})
                            ], id='stats-results-output', className="mt-4")
                            
                        ], id="collapse_stats", className="mt-3"),
                    ])
                ], className='mb-4'),

            ], className='col-lg-9 col-md-12 ps-4'),

        ], className="row"),
    ], className="container-fluid py-4"),
], id='analysis-layout')