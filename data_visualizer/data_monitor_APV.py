
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output,State
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import flask
from os import walk
import configparser
import sys
import ROOT as R
import log_loader_time
import os
import json
import glob2
import time

R.gROOT.SetBatch(1)
config = configparser.ConfigParser()
config.read("../config.ini")
try:
    data_folder=config["GLOBAL"].get("CIVETTA_APV_folder")
except KeyError as E:
    print (f"{E}Missing or partial configration file, restore it.")
    sys.exit(1)

import plotly.io as pio


# dirname = os.path.dirname(__file__)
# external_stylesheets = os.path.join(dirname, 'apv_style.css')
# print (external_stylesheets)
# external_stylesheets = "apv_style.css"
external_stylesheets = [dbc.themes.LUX]
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__) # define flask app.server
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets, title="Data visual planars - APV")

colorscale=[[0.0, "rgba(255,255,255,0.0)"],
                [0.0000000000011111, "#0d0887"],
                [0.1111111111111111, "#46039f"],
                [0.2222222222222222, "#7201a8"],
                [0.3333333333333333, "#9c179e"],
                [0.4444444444444444, "#bd3786"],
                [0.5555555555555556, "#d8576b"],
                [0.6666666666666666, "#ed7953"],
                [0.7777777777777778, "#fb9f3a"],
                [0.8888888888888888, "#fdca26"],
                [1.0, "#f9d221"]]

no_data_template = go.layout.Template()
no_data_template.layout.annotations = [
    dict(
        name="no_data watermark",
        text="NO DATA",
        textangle=-30,
        opacity=0.1,
        font=dict(color="black", size=100),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False
    )
]

avaible_runs=[]

avaible_runs.sort()
fig_0 = go.Figure()
fig_1 = go.Figure()
fig_2 = go.Figure()
fig_3 = go.Figure()

app.layout = html.Div(children=[
    html.H1(children='Planar setup data visualization (APV) '),
    
    html.Div([
    html.H6("Run"),
    dcc.Dropdown(
        id='sel_run',
        options=[{'label': i, 'value': i} for i in avaible_runs],
        value=''
        )

    ], style={'width': '10%', 'display': 'inline-block'}),
    
    
    html.Div([
    html.H6("Plot type"),
    dcc.Dropdown(
       id='plot_opt',
       options=[
            {"label": "------ Single hits -----", "value" : "", "disabled":True},
            {"label":"Charge vs time","value":"Charge vs time"},
            {"label":"Strip X","value":"Strip x"},
            {"label":"Strip Y","value":"Strip y"},
            {"label": "------- Clusters ------", "value": "", "disabled":True},
            {"label": "Charge", "value": "Charge"},
            {"label": "Cluster pos X", "value": "cl pos x"},
            {"label": "Cluster pos Y", "value": "cl pos y"},
            {"label": "Cluster time", "value": "cluster time"},

       ],
       value='Charge vs time'
        ),
    ], style={'width': '20%', 'display': 'inline-block', 'padding':'10px'}),
    

    html.Br(),
    html.Button('Plot', id='plot_but', n_clicks=0, className="btn btn-primary"),
    # html.Button('Elaborate data', id='elab_but', n_clicks=0),


    html.Br(),

#     html.Div(id='available_runs', style={'display': 'none'}),

#     html.Div(id='available_subs', style={'display': 'none'}),

    
    html.Div([

    html.H6("Planar 0"),
    dcc.Graph(
        id='data_visual_0',
        figure=fig_0,
        responsive = "auto"
    )], style={'width': '50%', 'display': 'inline-block'}),
    
    html.Div([

    html.H6("Planar 1"),
    dcc.Graph(
        id='data_visual_1',
        figure=fig_1,
        responsive = "auto"
    )], style={'width': '50%', 'display': 'inline-block'}),
    
    html.Div([

    html.H6("Planar 2"),
    dcc.Graph(
        id='data_visual_2',
        figure=fig_2,
        responsive = "auto"
    )], style={'width': '50%', 'display': 'inline-block'}),
    
    html.Div([

    html.H6("Planar 3"),
    dcc.Graph(
        id='data_visual_3',
        figure=fig_3,
        responsive = "auto"
    )], style={'width': '50%', 'display': 'inline-block'}),


])




@app.callback(
    Output('data_visual_0', 'figure'),
    Output('data_visual_1', 'figure'),
    Output('data_visual_2', 'figure'),
    Output('data_visual_3', 'figure'),
    Input('plot_but', 'n_clicks'),
    State('sel_run', 'value'),
    State('plot_opt', 'value')
)
def update_graph(n_clicks, sel_run, plot_opt):
    if n_clicks==0:
        return [fig_0, fig_1, fig_2, fig_3]
    fig_list=[]
    ## Hit plot
    if plot_opt=="Charge vs time":
        hit_pd_APV=pd.read_pickle(os.path.join(data_folder,f"APV_run_{sel_run}_hits.gzip"), compression="gzip")
        for planar in range (0,4):
            hit_pd_APV_c=hit_pd_APV[hit_pd_APV.GemHit_plane==planar]
            fig_list.append(charge_vs_time_plot(hit_pd_APV_c))

    if plot_opt=="Strip x":
        hit_pd_APV=pd.read_pickle(os.path.join(data_folder,f"APV_run_{sel_run}_hits.gzip"), compression="gzip")
        for planar in range (0,4):
            hit_pd_APV_c=hit_pd_APV[hit_pd_APV.GemHit_plane==planar]
            fig_list.append(strip_plot(hit_pd_APV_c, "x"))

    if plot_opt=="Strip y":
        hit_pd_APV=pd.read_pickle(os.path.join(data_folder,f"APV_run_{sel_run}_hits.gzip"), compression="gzip")
        for planar in range (0,4):
            hit_pd_APV_c=hit_pd_APV[hit_pd_APV.GemHit_plane==planar]
            fig_list.append(strip_plot(hit_pd_APV_c, "y"))
    ### Clusters plot
    if plot_opt=="Charge":
        cl_pd_apv=pd.read_pickle(os.path.join(data_folder,f"APV_run_{sel_run}_cl.gzip"), compression="gzip")
        for planar in range (0,4):
            cl_pd_apv_c=cl_pd_apv[cl_pd_apv.GemCluster1d_plane==planar]
            fig_list.append(charge_plot_clusters(cl_pd_apv_c))
    if plot_opt=="cl pos x":
        cl_pd_apv=pd.read_pickle(os.path.join(data_folder,f"APV_run_{sel_run}_cl.gzip"), compression="gzip")
        for planar in range (0,4):
            cl_pd_apv_c=cl_pd_apv[cl_pd_apv.GemCluster1d_plane==planar]
            fig_list.append(pos_plot_clusters(cl_pd_apv_c, strip="x"))

    if plot_opt=="cl pos y":
        cl_pd_apv=pd.read_pickle(os.path.join(data_folder,f"APV_run_{sel_run}_cl.gzip"), compression="gzip")
        for planar in range (0,4):
            cl_pd_apv_c=cl_pd_apv[cl_pd_apv.GemCluster1d_plane==planar]
            fig_list.append(pos_plot_clusters(cl_pd_apv_c, strip="y"))

    if plot_opt=="cluster time":
        cl_pd_apv=pd.read_pickle(os.path.join(data_folder,f"APV_run_{sel_run}_cl.gzip"), compression="gzip")
        for planar in range (0,4):
            cl_pd_apv_c=cl_pd_apv[cl_pd_apv.GemCluster1d_plane==planar]
            fig_list.append(cluster_time_plot(cl_pd_apv_c))
    for fig in fig_list:
        fig.update_layout(template="seaborn")
    return fig_list


@app.callback(
    Output('sel_run', 'options'),
    Input('sel_run', 'value'),

)
def update_list(sel_run):
    avaible_runs=[]
    for file, (run_n, ) in glob2.iglob(os.path.join(data_folder,"APV_run_*_hits.gzip"), with_matches=True):
       avaible_runs.append(run_n)
    avaible_runs.sort()
    return [{'label': i, 'value': i} for i in avaible_runs]




## Plot functions
def charge_vs_time_plot(hit_pd_APV):
    fig = px.density_heatmap(hit_pd_APV, x="GemHit_time", y="GemHit_q",
                             title="Charge vs time",
                             marginal_x="histogram",
                             marginal_y="histogram",
                             color_continuous_scale=colorscale,
                             nbinsx=150,
                             nbinsy=150)
    fig.update_layout(
        xaxis_title="Time ",
        yaxis_title="Charge [adc]",
        height=800
    )
    # fig.update_xaxes(range=[1300, 1600])
    # fig.update_yaxes(range=[0, 60])
    return fig

def strip_plot(hit_pd_APV, strip):
    if strip=="x":
        view=0
    else:
        view=1
    hit_pd_APV=hit_pd_APV[hit_pd_APV.GemHit_view==view]
    fig = px.histogram(hit_pd_APV, x="GemHit_strip",
                             title="Strip {}".format(strip),
                       nbins=128,
                       color_discrete_sequence=['indianred'])
    fig.update_layout(
        xaxis_title="Strip ",
        yaxis_title="Counts ",
        height=800
    )
    # fig.update_xaxes(range=[1300, 1600])
    # fig.update_yaxes(range=[0, 60])
    return fig

def charge_plot_clusters(cl_pd_APV):
    fig = px.histogram(cl_pd_APV, x="GemCluster1d_q",
                             title="Cluster charge",
                       color_discrete_sequence=['indianred'])
    fig.update_layout(
        xaxis_title="Charge [adc] ",
        yaxis_title="Counts ",
        height=800
    )
    # fig.update_xaxes(range=[1300, 1600])
    # fig.update_yaxes(range=[0, 60])
    return fig


def pos_plot_clusters(cl_pd_APV, strip):
    if strip=="x":
        view=0
    else:
        view=1
    cl_pd_APV = cl_pd_APV[cl_pd_APV.GemCluster1d_view == view]
    fig = px.histogram(cl_pd_APV, x="GemCluster1d_pos",
                             title="Cluster charge",
                             nbins=800,
                       color_discrete_sequence=['indianred'])
    fig.update_layout(
        xaxis_title=f"Cluster pos {strip} ",
        yaxis_title="Counts ",
        height=800
    )
    # fig.update_xaxes(range=[1300, 1600])
    # fig.update_yaxes(range=[0, 60])
    return fig

def cluster_time_plot(cl_pd_APV):
    cl_pd_APV=cl_pd_APV[cl_pd_APV.GemCluster1d_t0>0]
    fig = px.histogram(cl_pd_APV, x="GemCluster1d_t0",
                             title="Cluster charge",
                             nbins=500,
                       color_discrete_sequence=['indianred'])
    fig.update_layout(
        xaxis_title=f"Cluster t0 ",
        yaxis_title="Counts ",
        height=800
    )
    # fig.update_xaxes(range=[1300, 1600])
    # fig.update_yaxes(range=[0, 60])
    return fig
if __name__ == '__main__':
    debug = True
    app.run_server(debug=True, host="192.168.1.232")
