
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import flask
from os import walk
import configparser
import sys

config=configparser.ConfigParser()
config.read("../config.ini")
try:
    data_folder=config["GLOBAL"].get("data_folder")
    signal_upper_limit=config["GLOBAL"].getint("signal_window_upper_limit")
    signal_lower_limit=config["GLOBAL"].getint("signal_window_lower_limit")
except KeyError as E:
    print (f"{E}Missing or partial configration file, restore it.")
    sys.exit(1)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__) # define flask app.server
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server,title="Event visualizer")
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

avaible_runs=[]

# count=77
x_lines = [0,128,128,0,0]
y_lines = [0,0,128,128,0]
z_lines = [0,0,0,0,0]


trace0 = go.Scatter3d(
    mode="lines",
    x=x_lines,
    y=y_lines,
    z=z_lines,
    name="planar 0",
    opacity=0.8,
    showlegend=True,hovertemplate=None,
    hoverinfo='skip'
#     mode="lines",
#     hovertext="planar 1"

)

z_lines=[1,1,1,1,1]
trace1 = go.Scatter3d(
    mode="lines",
    x=x_lines,
    y=y_lines,
    z=z_lines,
    name="planar 1",
    opacity=0.8,
    showlegend=True,
    hoverinfo='skip'
#     mode="lines",
#     hovertext="planar 1"

)
z_lines=[2,2,2,2,2]
trace2 = go.Scatter3d(
    mode="lines",
    x=x_lines,
    y=y_lines,
    z=z_lines,
    name="planar 2",
    opacity=0.8,
    showlegend=True,
    hoverinfo='skip'

#     mode="lines",
#     hovertext="planar 2"

)
z_lines=[3,3,3,3,3]
trace3 = go.Scatter3d(
    mode="lines",
    x=x_lines,
    y=y_lines,
    z=z_lines,
    name="planar 3",
    opacity=0.8,
    showlegend=True,
    hoverinfo='skip'

#     mode="lines",
#     hovertext="planar 3"

)
fig = go.Figure(data=[trace0,trace1,trace2,trace3])

for (dirpath, dirnames, filenames) in walk(data_folder+"/raw_root"):
            for dirname in dirnames:
                if dirname.isdigit():
                    avaible_runs.append(int(dirname))
app.layout = html.Div(children=[
    html.H1(children='Planar setup clusters visualization '),
    # html.Div(children='''
    #     A simple visualizer fot the data
    # '''),
    html.Div([
        html.H6("Run"),
        dcc.Dropdown(
            id='sel_run',
            options=[{'label': i, 'value': i} for i in avaible_runs],
            value='15'
        )

    ], style={'width': '10%', 'display': 'inline-block'}),

    html.Div([
    html.H6("Subrun"),
    dcc.Dropdown(
       id='sel_subrun',
       options=[],
       value='0'),
    ], style={'width': '25%', 'display': 'inline-block'}),

    html.Div([
    html.H6("Count"),
    dcc.Dropdown(
        id='sel_count'
        # options=[{'label': i, 'value': i} for i in available_counts],
        # value='7'
        )

    ], style={'width': '25%', 'display': 'inline-block'}),

    html.Br(),
    html.Div(id='available_subs', style={'display': 'none'}),
    html.Div(id='available_counts', style={'display': 'none'}),
    html.Div([

    dcc.Graph(
        id='data_visual',
        figure=fig,
        responsive = "auto"
    )], style={'width': '74%', 'display': 'inline-block'}),
    html.Div([
    dcc.Markdown('''
    The plot shows the data from the planar setup @ Ferrara.

    The plot shows the planar chambers positions, the single hits on each views ('x') and the 2-D cluster positions (points).

    The 2-D cluster information is built clusterizing the data in each view and then taking the cluster with more charge for each view and planar.

    To display the data, select a subrun and a trigger count. Only counts with at least one cluster (i.e. one cluster per view in at least one planar) can be shown.

    The hits time is "-(l1ts_min_tcoarse-1370)" (where the signal region begins), in ns

    ''')],
    style={'width': '24%', 'display': 'inline-block', 'float':'right'})
])

@app.callback(
    Output('data_visual', 'figure'),
    Input('sel_count', 'value'),
    State('sel_subrun', 'value'),
    State('sel_run', 'value'),

)

def update_graph(sel_count,sel_subrun,sel_run):
    data_pd = pd.read_pickle("{}/raw_root/{}/hit_data.pickle.gzip".format(data_folder, sel_run), compression="gzip")
    cluster_pd_2D = pd.read_pickle("{}/raw_root/{}/cluster_pd_2D.pickle.gzip".format(data_folder, sel_run), compression="gzip")
    tracks_pd= pd.read_pickle("{}/raw_root/{}/tracks_pd_1D.pickle.gzip".format(data_folder, sel_run), compression="gzip")

    cluster_pd_cut = cluster_pd_2D[(cluster_pd_2D["count"]==sel_count) & (cluster_pd_2D["subrun"]==sel_subrun)]
    data_pd_cut = data_pd[(data_pd.l1ts_min_tcoarse>signal_lower_limit) & (data_pd.l1ts_min_tcoarse<signal_upper_limit) & (data_pd["count"]==sel_count) & (data_pd["subRunNo"]==sel_subrun)]
    tracks_pd_cut = tracks_pd[(tracks_pd["count"] == sel_count) & (tracks_pd["subrun"] == sel_subrun)]
    x_array=[]
    y_array=[]
    if len(tracks_pd_cut.x_fit[tracks_pd_cut.x_fit.notna()])>0:
        x_array= [(tracks_pd_cut.x_fit[tracks_pd_cut.x_fit.notna()].values[0][1] + tracks_pd_cut.x_fit[tracks_pd_cut.x_fit.notna()].values[0][0]*z) / 0.0650 for z in [0,10,20,30]]
    if len(tracks_pd_cut.y_fit[tracks_pd_cut.y_fit.notna()]) > 0:
        y_array= [(tracks_pd_cut.y_fit[tracks_pd_cut.y_fit.notna()].values[0][1] + tracks_pd_cut.y_fit[tracks_pd_cut.y_fit.notna()].values[0][0]*z) / 0.0650 for z in [0,10,20,30]]
    track_x = go.Scatter3d(mode="lines",x=x_array, y=np.zeros(4), z=[0,1,2,3], name= "track X")
    track_y = go.Scatter3d(mode="lines",x=np.zeros(4), y=y_array, z=[0,1,2,3], name= "track Y")

    fig_hits = go.Scatter3d(mode='markers',x=data_pd_cut.strip_x, y=data_pd_cut.strip_y, z=data_pd_cut.planar,marker_symbol="x",
                              customdata=np.stack((data_pd_cut.charge_SH, (data_pd_cut.l1ts_min_tcoarse-1370)*(-6.25)), axis=1),
                              marker=dict(size=3),name ="Single hits",
                              hovertemplate='pos_x: %{x:.2f}, pox_y%{y:.2f} <br>Charge: %{customdata[0]:.2f}fC <br>Time: %{customdata[1]:.2f}' )
    scatter = go.Scatter3d(mode='markers', x=cluster_pd_cut.cl_pos_x, y=cluster_pd_cut.cl_pos_y, z=cluster_pd_cut.planar,
                             marker=dict(size=4,color=cluster_pd_cut.cl_charge,colorscale="Viridis",cmax=100,cmin=0, showscale=True,
                             colorbar=dict(title="Cluster charge [fC]"),
                                        ),
                             name ="Clusters 2D",
                             customdata=cluster_pd_cut.cl_size_x+cluster_pd_cut.cl_size_y,
                             hovertemplate='pos_x: %{x:.2f}, pox_y%{y:.2f} <br>Charge: %{marker.color:.2f} <br> Size (x+y): %{customdata}',
                             showlegend=True)

    fig= go.Figure(data=[scatter,trace0,trace1,trace2,trace3,fig_hits])
    fig.add_traces([track_y,track_x])
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    camera = dict(
        up=dict(x=1, y=0, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=-2, z=2)
    )
    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=5, range=[-10,138],),
                          yaxis = dict(nticks=5, range=[-10,138],),
                          zaxis = dict(nticks=5, range=[3.5,-0.5],),),
        height=600,
        width=1200,
        scene_camera=camera,
        scene_aspectmode='manual',
        scene_aspectratio = dict(x=0.897,y=0.897,z=4)   )
        # fig.add_trace(trace0)
    return fig

@app.callback(
    Output('sel_count', 'options'),
    Input('sel_subrun', 'value'),
    State('sel_run', 'value')

)
def update_list_count(sel_subrun, sel_run):
    cluster_pd_2D = pd.read_pickle("{}/raw_root/{}/cluster_pd_2D.pickle.gzip".format(data_folder, sel_run), compression="gzip")
    available_counts = np.sort(cluster_pd_2D[cluster_pd_2D.subrun==sel_subrun]['count'].unique())
    return [ {'label':name, 'value':name} for name in available_counts]

@app.callback(
    Output('sel_subrun', 'options'),
    Input('sel_run', 'value'),
)
def update_list_subrun(sel_run):
    cluster_pd_2D = pd.read_pickle("{}/raw_root/{}/cluster_pd_2D.pickle.gzip".format(data_folder, sel_run), compression="gzip")
    available_subs = np.sort(cluster_pd_2D['subrun'].unique())
    return [ {'label':name, 'value':name} for name in available_subs]

@app.callback(
    Output('sel_run', 'options'),
    Input('sel_run', 'value'),
)
def update_list_run(sel_run):
    avaible_runs=[]
    for (dirpath, dirnames, filenames) in walk(data_folder + "/raw_root"):
        for dirname in dirnames:
            if dirname.isdigit():
                avaible_runs.append(int(dirname))
    avaible_runs.sort()
    return [ {'label':name, 'value':name} for name in avaible_runs]
if __name__ == '__main__':
    debug = True
    app.run_server(debug=debug)
