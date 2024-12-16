import base64
import io
import os
from urllib.parse import quote as urlquote
import time

import dash
from dash import set_props, DiskcacheManager
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import zipfile


import diskcache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)



def read_data(lines):
    # Identify the header line starting with 'Frequency'
    header_line_index = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('Frequency'):
            header_line_index = idx
            break
    if header_line_index is None:
        print(f"No header found in file {filename}")
        return None
    # Extract header and data
    header_line = lines[header_line_index].strip()
    columns = re.split(r'\s+', header_line.replace('\t', ' '))
    data_str = ''.join(lines[header_line_index + 1:])
    data = pd.read_csv(
        StringIO(data_str),
        sep=r'\s+',
        names=columns,
        engine='python'
    )
    
    if impedance_parameters:
        # Filter columns to include only specified impedance parameters and Frequency
        columns_to_include = ['Frequency'] + [param for param in impedance_parameters if param in data.columns]
        data = data[columns_to_include]
    else:
        # Include all columns except 'Frequency' as parameters
        columns_to_include = [col for col in data.columns if col != 'Frequency']
    
    return data


# UPLOAD_DIRECTORY = "/project/app_uploaded_files"
SAVE_DIRECTORY = "./saved"

# if not os.path.exists(UPLOAD_DIRECTORY):
#     os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
# server = Flask(__name__)
# app = dash.Dash(server=server)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], background_callback_manager=background_callback_manager)


# @server.route("/download/<path:path>")
# def download(path):
#     """Serve a file from the upload directory."""
#     return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


app.layout = html.Div(
    [
        html.H1("File Browser"),
        html.H2("Upload"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and drop or click to select a file to upload."]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        dbc.Progress(id="animated-progress-bar"),
        html.Div([(
            dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
            ) for page in dash.page_registry.values()
        ]),
        dash.page_container
    ],
    style={"max-width": "500px"},
)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
    background=True,
    prevent_initial_call=True,
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    # with zipfile.ZipFile(uploaded_filenames, 'r') as zip_ref:
    #     zip_ref.extractall(directory_to_extract_to)
    for contents in uploaded_file_contents:
        content_type, content_string = contents.split(',')
        try:
            decoded = base64.b64decode(content_string)
            archive = zipfile.ZipFile(io.BytesIO(decoded))
            for i in archive.infolist():
                print(i.filename)
                if not os.path.splitext(i.filename)[1] == ".txt":
                    continue
                base, file_name = os.path.split(i.filename)
                base, well = os.path.split(base)
                print(well, file_name)
                data = archive.read(i.filename)
                data = read_data(data)
                print(data)
        except Exception as e:
            print(e)
        break

    for i in range(1, 11, 1):
        set_props("animated-progress-bar", {'value': int(i*10), 'label': "{}%".format(int(i*10)),'animated': True, 'striped': True})
        time.sleep(2)

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            # save_file(name, data)
            print(name)
    

    # files = uploaded_files()
    # if len(files) == 0:
    #     return [html.Li("No files yet!")]
    # else:
    #     return [html.Li(file_download_link(filename)) for filename in files]


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)