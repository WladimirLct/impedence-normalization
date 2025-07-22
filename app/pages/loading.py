# File: pages/loading.py

import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

from callbacks.loading import *

dash.register_page(__name__)

layout = html.Div([
    dcc.Location(id='url-p2', refresh='callback-nav'),

    dcc.Store(id='norm_info', storage_type='local'),
    dcc.Store(id='load_info', storage_type='local'),

    dcc.Interval(id='load_interval', interval=200, n_intervals=0),
    dcc.Interval(id='norm_interval', interval=500, n_intervals=0),

    html.Div([
        html.P(
            "Chargement des données", 
            className="""
            h1 text-center mb-5
        """),
        html.Div([
            html.P(children=[
                html.Span('0/0', id='file_load_idx', className='fw-bold'),
                ' fichiers chargés'
            ]),
            html.P(children=[
                html.Span('0/8', id='file_norm_idx', className='fw-bold'),
                html.Span(' étape de normalisation', id='file_norm_step')
            ])
        ], className="bg-light p-5 rounded")
    ], className="mx-auto", style={
        'width': '500px',
    }),
], style={
    'height': '90vh'
}, className='d-flex flex-column align-items-center justify-content-center',
id='loading-layout')