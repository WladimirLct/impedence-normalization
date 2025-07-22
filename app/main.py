import os
import sys
import signal
import shutil

import webbrowser
from threading import Timer

from dash import Dash
import dash_bootstrap_components as dbc
import plotly.io as pio

import kaleido
kaleido.get_chrome_sync()

pio.templates.default = "plotly_white"

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)
app._favicon = ("assets/favicon.co")

app.layout = []
port = 8050

if os.path.exists('tmp'):
    shutil.rmtree('tmp')  # This removes directory and all contents
os.makedirs('tmp')

if not os.path.exists('normalizations'):
    os.makedirs('normalizations')

normalizations = sorted(os.listdir('normalizations'))
if len(normalizations) > 20:
    for norm in normalizations[:-20]:
        os.remove(os.path.join('normalizations', norm))
        print(f"Removed old normalization: {norm}")

def open_browser():
	webbrowser.open_new("http://localhost:{}".format(port))

def signal_handler(sig, frame):
    print('Gracefully shutting down...')
    # Add cleanup code here
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    Timer(1, open_browser).start();
    app.run(debug=False, port=port)