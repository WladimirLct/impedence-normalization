# File: callbacks/analysis_page/groups.py

import uuid

from dash import dcc, callback, Output, Input, State, no_update, ctx , html
from dash.dependencies import ALL
import dash_bootstrap_components as dbc

from utils.groups import obj_wells_groups, get_list_wells_groups, get_list_wells, add_wells_groups, get_wells_not_in_group

from config.cache import cache
from config.options import colors

@callback(
    Output('input_name_group', 'value', allow_duplicate=True),
    Output('select_wells', 'value', allow_duplicate=True),
    Output('input_concentration', 'value', allow_duplicate=True),
    Output('select_unit', 'value', allow_duplicate=True),
    Output('label_group', 'value', allow_duplicate=True),
    Output('well_groups_data', 'data', allow_duplicate=True),
    Output('groups_enabled', 'data', allow_duplicate=True),
    Output('list_well_not_in_group', 'data', allow_duplicate=True),
    #
    Input('add-groups', 'n_clicks'),
    State('input_name_group', 'value'),
    State('select_wells', 'value'),
    State('label_group', 'value'),
    State('input_concentration', 'value'),
    State('select_unit', 'value'),
    State('well_groups_data', 'data'),
    State('wells', 'data'),
    prevent_initial_call=True,
)
# Callback to add a new group of wells
def add_groups(n_clicks, input_name_group, select_wells, label_group, input_concentration, select_unit, well_groups_data, wells_data):
    if n_clicks is None:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # Validate inputs
    if not input_name_group or len(select_wells) <= 0 or input_concentration is None or not select_unit or not label_group:
        print("Invalid group data")
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    new_group = obj_wells_groups(
        name=input_name_group.strip(),
        concentration=input_concentration,
        label=label_group,
        unit=select_unit,
        wells=select_wells
    )
    new_group['id'] = str(uuid.uuid4())

    list_groups = list(well_groups_data) if well_groups_data else []
    list_groups.append(new_group)

    list_wells_not_in_group = get_wells_not_in_group(wells_data, list_groups)

    return "", "", None, None, None, list_groups, True, list_wells_not_in_group


@callback(
    Output('well-groups', 'children'),
    #
    Input('well_groups_data', 'data'),
)
# Callback to render the well groups
def render_well_groups(well_groups_data):

    well_groups_data = well_groups_data or []

    if len(well_groups_data) < 1:
        return "Aucun groupe de puits défini."

    components = []
    for i, group in enumerate(well_groups_data):
        wells = ", ".join(group.get("wells", []))
        group_card = dbc.Card([
            dbc.CardBody([
                html.Div(f"{group['name']} ({group['concentration']} {group['unit']})", className="fw-bold mb-1"),
                dbc.Row([
                    dbc.Col(html.Span(group['label'], className=f"badge bg-{colors[group['label']]} text-dark"), width=8),
                    dbc.Col(
                        dbc.Button("Supprimer", id={'type': 'delete-group-btn', 'id': group["id"]}, color='danger', size='sm'),
                        width=4, className="d-flex align-items-center justify-content-end"
                    ),
                ], className="mb-1"),
                html.Div(f"{wells}", className="text-muted small"),
            ])
        ], className="mb-2")

        components.append(group_card)

    return components

@callback(
    Output('well_groups_data', 'data'),
    Output('groups_enabled', 'data', allow_duplicate=True),
    Output('list_well_not_in_group', 'data', allow_duplicate=True),
    #
    Input({'type': 'delete-group-btn', 'id': ALL}, 'n_clicks'),
    State('well_groups_data', 'data'),
    State('wells', 'data'),
    prevent_initial_call=True,
)
def delete_group(n_clicks_list, well_groups_data, wells_data):
    # print("Delete group callback triggered with n_clicks:", n_clicks_list)

    if not n_clicks_list or not any(n_clicks_list):
        return no_update, no_update, no_update

    triggered = ctx.triggered_id
    if not triggered:
        return no_update, no_update, no_update

    id_to_remove = triggered["id"]

    well_groups_data = [g for g in well_groups_data if g["id"] != id_to_remove]

    list_wells_not_in_group = get_wells_not_in_group(wells_data, well_groups_data)

    if len(well_groups_data) < 1:
        return [], False, list_wells_not_in_group

    return well_groups_data, True, list_wells_not_in_group

@callback(
    Output('select_wells', 'options', allow_duplicate=True),
    #
    Input('list_well_not_in_group', 'data'),
    State('wells', 'data'),
    prevent_initial_call=True,
)
def update_wells_dropdown(list_well_not_in_group, wells_data):

    if list_well_not_in_group is None :
        options = [{'label': well, 'value': well} for well in wells_data] 
    else:
        options = [{'label': well, 'value': well} for well in list_well_not_in_group] 

    return options


@callback(
    Output('select-group-ref', 'options'),
    Output('select-group-comp', 'options', allow_duplicate=True),
    #
    Input('well_groups_data', 'data'),
    State('select-group-ref', 'value'),
    prevent_initial_call=True  # pour initialiser au démarrage
)
def update_group_dropdown_options(well_groups_data, selected_ref_group):
    if not well_groups_data:
        return [], []

    # Créer les options pour le groupe de référence
    ref_options = [{'label': f"{group['name']} ({group['concentration']} {group['unit']})", 'value': group['id']} for group in well_groups_data]
    conf_options = [{'label': f"{group['name']} ({group['concentration']} {group['unit']})", 'value': group['id']} for group in well_groups_data if group['id'] != selected_ref_group]

    return ref_options, conf_options

@callback(
    Output('select-group-comp', 'options', allow_duplicate=True),
    #
    Input('select-group-ref', 'value'),
    State('well_groups_data', 'data'),
    prevent_initial_call=True
)
def filter_comparison_options(selected_ref_group, well_groups_data):
    if not well_groups_data:
        return []

    options = [
        {'label': f"{group['name']} ({group['concentration']} {group['unit']})", 'value': group['id']} for group in well_groups_data if group['id'] != selected_ref_group
    ]
    return options