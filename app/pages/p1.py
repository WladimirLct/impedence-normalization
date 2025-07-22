# File: pages/p1.py

import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

from callbacks.p1 import *
from config.cache import cache

dash.register_page(__name__, path='/')

layout = html.Div([
    dcc.Location(id='url-p1', refresh='callback-nav'),
    html.Div([
        html.P(
            "Projet YP-2425-35", 
            className="""
            h1 text-center mb-5
        """),
        html.Div([
            dbc.Row([
                html.P(
                    "Entrez le chemin d'accès dans lequel se trouve l'expérience à analyser. Le chemin doit pointer vers un dossier dans lequel se trouvent tous les puits.",
                    className="text-muted"
                )
            ]),
            dbc.Row([
                dbc.Input(
                    type='text',
                    placeholder="Chemin d'accès",
                    required=True,
                    className="col",
                    id='folder-path',
                ),
                dbc.Button('Charger', id='folder-btn', className="col-3"),
                html.Div(
                    dbc.Spinner(html.Div(id="loading-output", className="mt-4"), color="primary"),
                    className="col-3 d-none",
                    id='spinner-div',
                )
            ], className="row")
        ], className="bg-light p-5 rounded")
    ], className="mx-auto", style={
        'width': '500px',
    }),
    html.Div([
        html.P(
            id='data-info',
            className='text-muted'
        ),
        html.Div(
            dash_table.DataTable(
                id='data-table',
                page_size=20,
                style_cell={'textAlign': 'left'},
            ), className="overflow-auto", style={
                'max-height': '720px' 
            }
        )
    ], className="mt-5", style={
        'width': '800px',
    })
], style={
    'height': '90vh'
}, className='d-flex flex-column align-items-center justify-content-center',
id='p1-layout')