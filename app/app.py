# app.py
import dash
from dash import html, dcc, Input, Output
from app_instance import app  # Import app and cache from app_instance
import webbrowser
from threading import Timer  # Use Timer to delay the browser opening

# Define the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='session-store', storage_type='session'),
    html.Div(id='page-content')
])

# Import page layouts and callbacks
from main_page import main_layout, register_main_callbacks
from stat_analysis import stats_layout, register_stats_callbacks

# Update `validation_layout` to include all pages and the 'return-button'
app.validation_layout = html.Div([
    app.layout,
    main_layout,
    stats_layout,
    html.Button(id='return-button', style={'display': 'none'}),
])

# Callback to control page navigation
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/stats':
        return stats_layout
    else:
        return main_layout

# Unified navigation callback
@app.callback(
    Output('url', 'pathname'),
    Input('nav-to-stats', 'n_clicks'),
    prevent_initial_call=True
)
def navigate_to_stats(n_clicks):
    if n_clicks:
        return '/stats'
    else:
        raise dash.exceptions.PreventUpdate

# Import and register page-specific callbacks
from main_page import register_main_callbacks
register_main_callbacks(app)  # Callbacks for the main page

register_stats_callbacks(app)  # Callbacks for the stats page

browser_opened = False  # Global flag to track if the browser has been opened

def open_browser():
    global browser_opened
    if not browser_opened:  # Check if the browser has already been opened
        webbrowser.open_new("http://127.0.0.1:8050/")
        browser_opened = True  # Set the flag to True

if __name__ == '__main__':
    
    import multiprocessing
    multiprocessing.freeze_support()  # <-- Enables proper behavior in frozen apps

    from threading import Timer
    Timer(1, open_browser).start()  # This will run only once in the main process
    
    app.run_server(debug=False)