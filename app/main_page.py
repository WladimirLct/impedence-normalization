import os
import pandas as pd
import plotly
import plotly.graph_objects as go

from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import dash.exceptions
import dash

from dash.dependencies import ALL  # Pour les callbacks dynamiques

from app_instance import app, data_cache  # Import de l'instance app et du cache
import uuid

from io import StringIO
import time



# Configuration
CONFIG = {
    "impedance_params": ['Param0', 'Param1', 'AbsZ', 'RealZ', 'ImagZ', 'PhaseZ'],
    "group_size": 3,
    "date_format": '%Y-%m-%d %H:%M:%S',
    "save_dir": "./analysis_results",
    "colorscale": plotly.colors.qualitative.Dark24
}

# Création du répertoire de sauvegarde s'il n'existe pas
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# Mise en page principale
main_layout = dbc.Container([
    # Stockages pour la session et pour stocker les groupes définis manuellement
    dcc.Store(id='group-assignments', storage_type='session'),
    dcc.Store(id='session-store', storage_type='session'),

    dbc.Row([
        # Colonne de gauche : Data Management, Group Wells et Data Export
        dbc.Col([
            # Carte Data Management
            dbc.Card([
                dbc.CardHeader("Data Management", className="h5"),
                dbc.CardBody([
                    dbc.InputGroup([
                        dbc.Input(
                            id="input-path",
                            placeholder="Entrez/coller le chemin du répertoire",
                            className="rounded-start"
                        ),
                        dbc.Button(
                            "Analyze",
                            id="analyze-button",
                            color="primary",
                            className="rounded-end"
                        )
                    ], className="mb-3"),
                    html.Small(
                        id="loading-output",
                        className="text-muted d-block text-center",
                        children="Ready"
                    )
                ])
            ], className="mb-4"),

            # Carte Group Wells (affichée si l'option est activée)
            dbc.Card([
                dbc.CardHeader("Group Wells", className="h5"),
                dbc.CardBody([
                    # Chaque groupe comporte deux dropdowns : puits et type
                    html.Div(id="group-wells-container", children=[]),
                    dbc.Button("Add Wells Group", id="add-group-button", color="primary", size="sm", className="mt-2")
                ])
            ], className="mb-4", id="group-wells-card", style={"display": "none"}),

            # Carte Data Export
            dbc.Card([
                dbc.CardHeader("Data Export", className="h5"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Frequencies :", className="mb-2"),
                            dcc.Dropdown(
                                id="download-frequency-selector",
                                multi=True,
                                placeholder="Sélectionnez les fréquences...",
                                className="dropdown-primary"
                            )
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Wells :", className="mb-2"),
                            dcc.Dropdown(
                                id="download-well-selector",
                                multi=True,
                                placeholder="Sélectionnez les puits ou groupes...",
                                className="dropdown-primary"
                            )
                        ], md=6)
                    ], className="g-2 mb-3"),
                    dbc.Button("Download CSV Bundle", id="download-button", color="success", className="w-100", outline=True),
                    dcc.Download(id="download-csv"),
                    dbc.Button("Statistical Analysis", id="nav-to-stats", color="secondary", className="w-100 mt-3", outline=True)
                ])
            ])
        ], md=3, className="pe-3"),

        # Colonne de droite : Visualisation et Contrôles
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Visualization Controls", className="h5"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Normalization Method :", className="mb-2"),
                            dcc.Dropdown(
                                id="normalization-method",
                                options=[
                                    {'label': 'Absolute Z', 'value': 'AbsZ'},
                                    {'label': 'T0 Normalised', 'value': 'AbsZ_t0'},
                                    {'label': 'Max Normalized', 'value': 'AbsZ_max'},
                                    {'label': 'Min-Max Scaled', 'value': 'AbsZ_min_max'},
                                    # Options personnalisées
                                    {'label': 'Treated/Control Normalized', 'value': 'CustomNorm1'},
                                    {'label': 'Infected Differential Normalized', 'value': 'CustomNorm2'}
                                ],
                                value='AbsZ',
                                clearable=False,
                                className="mb-3"
                            )
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Active Frequencies :", className="mb-2"),
                            dcc.Dropdown(
                                id="frequency-selector",
                                multi=True,
                                clearable=False,
                                className="mb-3"
                            )
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Display Settings :", className="mb-2"),
                            dbc.Checklist(
                                options=[{"label": " Group Wells", "value": "group-wells"}],
                                value=[],
                                id="view-options",
                                switch=True,
                                className="mb-3"
                            )
                        ], md=4)
                    ], className="g-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Std Multiplier :", className="mb-2"),
                            dcc.Slider(
                                id="std-scale",
                                min=0,
                                max=3.0,
                                step=0.2,
                                value=1.0,
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Data Density :", className="mb-2"),
                            dcc.Slider(
                                id="data-resolution",
                                min=1,
                                max=100,
                                step=1,
                                value=1,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=6)
                    ], className="g-3")
                ])
            ], className="mb-4"),
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        id="impedance-plot",
                        config={"displayModeBar": True},
                        style={"height": "65vh", "minHeight": "500px"},
                        className="border rounded-3"
                    )
                ])
            ]),
            # Ajout d'une nouvelle carte pour le champ de commentaire
            dbc.Card([
                dbc.CardHeader("Laissez un commentaire sur la visualisation"),
                dbc.CardBody([
                    dcc.Textarea(
                        id="visualization-comment",
                        placeholder="Entrez votre commentaire ici...",
                        style={"width": "100%", "height": "100px", "resize": "none"}
                    ),
                    dbc.Button("Soumettre", id="submit-comment", color="primary", className="mt-2"),
                    html.Div(id="comment-output", className="mt-3 text-success")
                ])
            ], className="mt-3")

        ], md=9, className="ps-3")
    ], className="g-4"),
    html.Div(id="dummy-output", style={"display": "none"})
], fluid=True, className="dbc bg-light")

# =============================
# CALLBACKS
# =============================

def register_main_callbacks(app):

    # --- Chargement des données ---
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
        if not n_clicks or n_clicks == 0:
            raise dash.exceptions.PreventUpdate
        if not path or not os.path.exists(path):
            return dash.no_update, dash.no_update, "Invalid directory path", dash.no_update
        session_id = str(uuid.uuid4())
        data_cache.clear()
        try:
            save_path = os.path.join(CONFIG["save_dir"], f"{os.path.basename(path)}.csv")
            if os.path.exists(save_path):
                df = pd.read_csv(save_path)
            else:
                df = load_wells(path, save_path, CONFIG["impedance_params"])
                df.to_csv(save_path, index=False)
            # Conversion explicite des colonnes d'intérêt en numérique
            cols_interest = ['AbsZ', 'AbsZ_t0', 'AbsZ_max', 'AbsZ_min_max', 'CustomNorm1', 'CustomNorm2']
            for col in cols_interest:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            data_cache[session_id] = df
            freq_options = [{'label': f"{frq:.2f} Hz", 'value': frq}
                            for frq in sorted(df['Frequency'].unique())]
            default_freq = [df['Frequency'].iloc[0]]
            return freq_options, default_freq, "Data loaded successfully", session_id
        except Exception as e:
            return dash.no_update, dash.no_update, f"Error: {str(e)}", dash.no_update

    # --- Ajout d'un dropdown pour saisir un groupe (avec type) ---
    @app.callback(
        Output("group-wells-container", "children"),
        Input("add-group-button", "n_clicks"),
        State("group-wells-container", "children")
    )
    def add_group_dropdown(n_clicks, children):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        if children is None:
            children = []
        new_index = len(children)
        new_group = html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label(f"Group {new_index + 1} - Wells"),
                    dcc.Dropdown(
                        id={'type': 'group-dropdown', 'index': new_index},
                        multi=True,
                        placeholder="Sélectionnez les puits..."
                    )
                ], width=8),
                dbc.Col([
                    dbc.Label("Type", className="mb-1"),
                    dcc.Dropdown(
                        id={'type': 'group-type', 'index': new_index},
                        options=[
                            {"label": "Infecté", "value": "infected"},
                            {"label": "Traité", "value": "treated"},
                            {"label": "Contrôle", "value": "control"}
                        ],
                        value="control",  # Valeur par défaut
                        clearable=False
                    )
                ], width=4)
            ], className="mb-2")
        ])
        children.append(new_group)
        return children

    # --- Mise à jour dynamique des options des dropdowns de groupe ---
    @app.callback(
        Output({'type': 'group-dropdown', 'index': ALL}, 'options'),
        Input({'type': 'group-dropdown', 'index': ALL}, 'value'),
        State("session-store", "data"),
        State({'type': 'group-dropdown', 'index': ALL}, 'id')
    )
    def update_group_dropdown_options(selected_values, session_id, ids):
        if not session_id:
            return dash.no_update
        df = data_cache.get(session_id)
        if df is None:
            return dash.no_update
        all_wells = sorted(df["Well"].unique())
        outputs = []
        used = set()
        # Trier les dropdowns par leur index
        ids_sorted = sorted(ids, key=lambda d: d['index'])
        for i, d in enumerate(ids_sorted):
            available = [w for w in all_wells if w not in used]
            outputs.append([{'label': w, 'value': w} for w in available])
            if selected_values and len(selected_values) > i and selected_values[i]:
                used.update(selected_values[i])
        return outputs

    # --- Mise à jour de la visualisation ---
    @app.callback(
        [Output("impedance-plot", "figure"),
         Output("group-assignments", "data")],
        [Input("session-store", "data"),
         Input("normalization-method", "value"),
         Input("frequency-selector", "value"),
         Input("view-options", "value"),
         Input("std-scale", "value"),
         Input("data-resolution", "value"),
         Input({'type': 'group-dropdown', 'index': ALL}, 'value'),
         Input({'type': 'group-type', 'index': ALL}, 'value')],
        [State("input-path", "value"),
         State("group-assignments", "data")]
    )
    def update_visualization(session_id, norm_method, frequencies, view_options,
                             std_scale, data_resolution, selected_groups, selected_types, path, stored_groups):
        if not session_id or not frequencies:
            return go.Figure(layout={'template': 'plotly_white'}), dash.no_update
        df = data_cache.get(session_id)
        if df is None:
            return go.Figure(layout={'template': 'plotly_white'}), dash.no_update
        filtered_df = df[df['Frequency'].isin(frequencies)]
        
        # Stocker la liste des groupes avec leur type pour Data Export
        group_assignments = []
        
        if 'group-wells' in view_options:
            if selected_groups and any(selected_groups):
                manual_groups = []
                # Si une méthode personnalisée est choisie
                if norm_method in ["CustomNorm1", "CustomNorm2"]:
                    if norm_method == "CustomNorm1":
                        treated, control = None, None
                        for wells, gtype in zip(selected_groups, selected_types):
                            if wells:
                                if gtype == "treated":
                                    treated = wells
                                elif gtype == "control":
                                    control = wells
                        if treated is not None and control is not None:
                            df_treated = filtered_df[filtered_df['Well'].isin(treated)]
                            df_control = filtered_df[filtered_df['Well'].isin(control)]
                            
                            # Calcul basé sur les colonnes de df, à t0
                            treated_mean = df_treated.groupby(['hours', 'Frequency'])['AbsZ'].mean().reset_index()
                            control_mean = df_control.groupby(['hours', 'Frequency'])['AbsZ'].mean().reset_index()
                            merged = pd.merge(treated_mean, control_mean, on=['hours','Frequency'], suffixes=('_treated', '_control'))
                            merged['Znorm'] = merged['AbsZ_treated'] / merged['AbsZ_control']
                            merged['Well'] = "Treated/Control"
                            
                            print("Valeurs de Znorm (CustomNorm1) :")
                            print(merged[['hours', 'Frequency', 'Znorm']].head())
                            
                            # Mise à jour de la DataFrame d'origine avec la colonne CustomNorm1
                            original_df = data_cache.get(session_id)
                            if original_df is not None:
                                df_updated = pd.merge(original_df, merged[['hours', 'Frequency', 'Znorm']], on=['hours', 'Frequency'], how='left')
                                df_updated.rename(columns={'Znorm': 'CustomNorm1'}, inplace=True)
                                data_cache[session_id] = df_updated
                            
                            fig = generate_plot(merged, "Znorm", std_scale, data_resolution)
                            group_assignments.append({"wells": treated, "type": "treated"})
                            group_assignments.append({"wells": control, "type": "control"})
                            return fig, group_assignments
                    elif norm_method == "CustomNorm2":
                        infected, treated, control = None, None, None
                        for wells, gtype in zip(selected_groups, selected_types):
                            if wells:
                                if gtype == "infected":
                                    infected = wells
                                elif gtype == "treated":
                                    treated = wells
                                elif gtype == "control":
                                    control = wells
                        if infected is not None and treated is not None and control is not None:
                            df_infected = filtered_df[filtered_df['Well'].isin(infected)]
                            df_treated = filtered_df[filtered_df['Well'].isin(treated)]
                            df_control = filtered_df[filtered_df['Well'].isin(control)]
                            infected_mean = df_infected.groupby(['hours', 'Frequency'])['AbsZ'].mean().reset_index()
                            treated_mean = df_treated.groupby(['hours', 'Frequency'])['AbsZ'].mean().reset_index()
                            control_mean = df_control.groupby(['hours', 'Frequency'])['AbsZ'].mean().reset_index()
                            merged = pd.merge(infected_mean, control_mean, on=['hours','Frequency'], suffixes=('_infected', '_control'))
                            merged = pd.merge(merged, treated_mean, on=['hours','Frequency'])
                            merged.rename(columns={'AbsZ': 'AbsZ_treated'}, inplace=True)
                            merged['Znorm'] = (merged['AbsZ_infected'] - merged['AbsZ_control']) / (merged['AbsZ_infected'] - merged['AbsZ_treated'])
                            merged['Well'] = "Infected Differential"
                            
                            print("Valeurs de Znorm (CustomNorm2) :")
                            print(merged[['hours', 'Frequency', 'Znorm']].head())
                            
                            # Mise à jour de la DataFrame d'origine avec la colonne CustomNorm2
                            original_df = data_cache.get(session_id)
                            if original_df is not None:
                                df_updated = pd.merge(original_df, merged[['hours', 'Frequency', 'Znorm']], on=['hours', 'Frequency'], how='left')
                                df_updated.rename(columns={'Znorm': 'CustomNorm2'}, inplace=True)
                                data_cache[session_id] = df_updated
                            
                            fig = generate_plot(merged, "Znorm", std_scale, data_resolution)
                            group_assignments.append({"wells": infected, "type": "infected"})
                            group_assignments.append({"wells": treated, "type": "treated"})
                            group_assignments.append({"wells": control, "type": "control"})
                            return fig, group_assignments
                else:
                    # Normalisation standard pour chaque groupe
                    manual_groups = []
                    for wells, gtype in zip(selected_groups, selected_types):
                        if wells:
                            group_assignments.append({"wells": wells, "type": gtype})
                            group_df = filtered_df[filtered_df['Well'].isin(wells)]
                            if not group_df.empty:
                                agg_df = group_df.groupby(['hours', 'Frequency'])[norm_method].mean().reset_index()
                                agg_df['Well'] = f"Group ({gtype}): " + ", ".join(wells)
                                manual_groups.append(agg_df)
                    if manual_groups:
                        aggregated_df = pd.concat(manual_groups, ignore_index=True)
                        fig = generate_plot(aggregated_df, norm_method, std_scale, data_resolution)
                        return fig, group_assignments
            # Aucun groupe défini manuellement, donc regroupement automatique
            groups_auto = create_well_groups(filtered_df)
            grouped_df = apply_group_averaging(filtered_df, groups_auto, norm_method)
            fig = generate_plot(grouped_df, norm_method, std_scale, data_resolution)
            return fig, group_assignments
        else:
            fig = generate_plot(filtered_df, norm_method, std_scale, data_resolution)
            return fig, dash.no_update

    # --- Mise à jour des options des menus déroulants de téléchargement ---
    @app.callback(
        [Output("download-frequency-selector", "options"),
         Output("download-well-selector", "options")],
        [Input("session-store", "data"),
         Input("group-assignments", "data")]
    )
    def update_download_options(session_id, group_assignments):
        if not session_id:
            return [], []
        df = data_cache.get(session_id)
        if df is None:
            return [], []
        freq_options = [{'label': f"{frq:.2f} Hz", 'value': frq} for frq in sorted(df['Frequency'].unique())]
        individual_wells = sorted(df['Well'].unique())
        well_options = [{'label': w, 'value': w} for w in individual_wells]
        if group_assignments:
            for i, group in enumerate(group_assignments):
                if group.get("wells"):
                    label = f"Group ({group['type']}): " + ", ".join(group["wells"])
                    well_options.append({'label': label, 'value': f"group_{i}"})
        return freq_options, well_options

    # --- Génération du téléchargement du CSV filtré ---
    @app.callback(
        Output("download-csv", "data"),
        Input("download-button", "n_clicks"),
        [State("session-store", "data"),
         State("download-frequency-selector", "value"),
         State("download-well-selector", "value"),
         State("group-assignments", "data")],
        prevent_initial_call=True
        )
    def generate_csv(n_clicks, session_id, frequencies, wells, group_assignments):
        if not session_id or not frequencies or not wells:
            raise dash.exceptions.PreventUpdate
        df = data_cache.get(session_id)
        if df is None:
            raise dash.exceptions.PreventUpdate
        final_wells = []
        for w in wells:
            if isinstance(w, str) and w.startswith("group_"):
                try:
                    idx = int(w.split("_")[1])
                    if group_assignments and idx < len(group_assignments):
                        final_wells.extend(group_assignments[idx]["wells"])
                except Exception:
                    continue
            else: 
                final_wells.append(w)
        final_wells = list(set(final_wells))
        filtered_df = df[df['Frequency'].isin(frequencies) & df['Well'].isin(final_wells)]
        if filtered_df.empty:
            raise dash.exceptions.PreventUpdate
        return dcc.send_data_frame(
            filtered_df.to_csv,
            filename=f"bioimpedance_data_{time.strftime('%Y%m%d-%H%M%S')}.csv",
            index=False
        )

    # --- Affichage/Masquage de la carte "Group Wells" ---
    @app.callback(
        Output("group-wells-card", "style"),
        Input("view-options", "value")
    )
    def toggle_group_wells_card(view_options):
        if view_options and "group-wells" in view_options:
            return {"display": "block"}
        else:
            return {"display": "none"}
        
    @app.callback(
        Output("comment-output", "children"),
        Input("submit-comment", "n_clicks"),
        State("visualization-comment", "value"),
        prevent_initial_call=True
    )
    def save_comment(n_clicks, comment):
        if not comment:
            return "Aucun commentaire soumis."
        return f"Commentaire soumis : {comment}"



# =============================
# FONCTIONS D'AIDE
# =============================

def filter_frequencies(df, frequencies):
    """Filtre le DataFrame pour ne conserver que les fréquences sélectionnées."""
    return df[df['Frequency'].isin(frequencies)]

def create_well_groups(df):
    """Crée des groupes automatiques de puits en fonction de la taille définie dans CONFIG."""
    wells = sorted(df["Well"].unique())
    grouped_wells = [wells[i:i + CONFIG["group_size"]] for i in range(0, len(wells), CONFIG["group_size"])]
    return grouped_wells

def apply_group_averaging(df, groups, norm_method):
    """Calcule la moyenne de la colonne d'intérêt pour chaque groupe de puits."""
    averaged_data = []
    for group in groups:
        group_df = df[df['Well'].isin(group)]
        group_mean = group_df.groupby(['hours', 'Frequency'])[norm_method].mean().reset_index()
        group_mean['Well'] = f"Group {', '.join(group)}"
        averaged_data.append(group_mean)
    return pd.concat(averaged_data, ignore_index=True)

def generate_plot(df, norm_method, std_scale=1.0, data_resolution=1):
    """Génère la figure Plotly en fonction des données, de la méthode de normalisation et des paramètres d'affichage."""
    fig = go.Figure()
    colors = plotly.colors.qualitative.Dark24

    for idx, well in enumerate(df['Well'].unique()):
        well_data = df[df['Well'] == well].sort_values('hours')

        if data_resolution > 1:
            well_data = well_data.iloc[::data_resolution]

        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=well_data['hours'],
            y=well_data[norm_method],
            name=well,
            mode='lines',
            line=dict(width=2, color=color),
        ))
        std = well_data[norm_method].std() * std_scale
        upper_bound = well_data[norm_method] + std
        lower_bound = well_data[norm_method] - std
        hex_color = color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        fillcolor = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)'

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