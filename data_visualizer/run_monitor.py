
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import flask
from os import walk
import configparser
import sys
import log_loader_time
import os
import json
import glob2

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
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server,title="Trigger stats")

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
for (dirpath, dirnames, filenames) in walk(data_folder+"/raw_dat"):
            for dirname in dirnames:
                dirname=dirname.replace("RUN_","")
                if dirname.isdigit():
                    if int(dirname)>130 and int(dirname)<5000:
                        avaible_runs.append(int(dirname))
avaible_runs.sort()
fig_0 = go.Figure()
fig_1 = go.Figure()
fig_2 = go.Figure()
fig_3 = go.Figure()


app.layout = html.Div(children=[
    html.H1(children='Tigger numbers '),

    html.Div([
    html.H6("Run"),
    dcc.Dropdown(
        id='sel_run',
        options=[{'label': i, 'value': i} for i in avaible_runs],
        value='107'
        )

    ], style={'width': '5%', 'display': 'inline-block'}),


    html.Br(),
    html.Button('Plot', id='plot_but', n_clicks=0),
    html.Br(),

#     html.Div(id='available_runs', style={'display': 'none'}),

#     html.Div(id='available_subs', style={'display': 'none'}),

    html.Div([
        html.H4("Table"),
        html.Table(id='display_sel_trigger',
                   children=''),
    ], style={'width': '20%', 'display': 'inline-block'}),

    html.Div([
    html.H6("Run vs trigger"),
    dcc.Graph(
        id='data_visual_0',
        figure=fig_0,
        responsive = "auto"
    )], style={'width': '80%', 'display': 'inline-block'}),

    html.Div([
    html.H6("Trigger vs subrun (selected run)"),
    dcc.Graph(
        id='data_visual_1',
        figure=fig_1,
        responsive = "auto"
    )], style={'width': '80%', 'display': 'inline-block'}),

    html.Div([
    html.H6("Place holder"),
    dcc.Graph(
        id='data_visual_2',
        figure=fig_2,
        responsive = "auto"
    )], style={'width': '50%', 'display': 'inline-block'}),

    html.Div([
    html.H6("Place holder"),
    dcc.Graph(
        id='data_visual_3',
        figure=fig_3,
        responsive = "auto"
    )], style={'width': '50%', 'display': 'inline-block'})

    ])



@app.callback(
    Output('data_visual_0', 'figure'),
    Output('data_visual_1', 'figure'),
    Output('data_visual_2', 'figure'),
    Output('data_visual_3', 'figure'),
    Output('display_sel_trigger', 'children'),
    Input('plot_but', 'n_clicks'),
    State('sel_run', 'value'),
)
def update_graph(n_clicks, sel_run):
    fig_list=[]
    trigger_pd=build_trigger_pd()
    fig_list.append(trigger_vs_run_plot(trigger_pd))
    fig_list.append(trigger_vs_subrun_plot(trigger_pd, sel_run))
    fig_list.append(trigger_vs_run_plot(trigger_pd))
    fig_list.append(trigger_vs_run_plot(trigger_pd))
    fig_list.append(generate_table(trigger_pd[trigger_pd.gemroc == 0].groupby("run").agg({"triggers":sum, "subrun": lambda x: x.nunique(), "time":min})))

    return fig_list

def generate_table(dataframe, max_rows=100):
    dataframe.insert(0, "run",dataframe.index)
    # dataframe=dataframe.drop(columns=['gemroc'])
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])
def build_trigger_pd():
    run_num = [int(num[0]) for file, num in (glob2.iglob(f"{data_folder}/raw_dat/RUN_*/", with_matches=True))]
    run_num = [num for num in run_num if num > 106]
    gemroc = []
    run = []
    subrun = []
    trigger = []
    time_end = []
    for run_n in run_num:
        for filename, sub in glob2.iglob(f"{data_folder}/raw_dat//RUN_{run_n}/ACQ_log_*", with_matches=True):
            with open(filename, 'r') as filelog:
                for line in filelog.readlines():
                    if "total" in line:
                        run.append(int(run_n))
                        subrun.append(int(sub[0]))
                        gemroc.append(int(line.split()[11]))
                        trigger.append(int(line.split()[-1]))
                        time_end.append(line.split("--")[0])
    trigger_pd = pd.DataFrame({"gemroc": gemroc, "run": run, "subrun": subrun, "triggers": trigger, "time":time_end})
    return trigger_pd


## Plot functions
def trigger_vs_run_plot(trigger_pd):

    fig = px.scatter(x=trigger_pd[trigger_pd.gemroc==0].groupby("run").sum().index, y=trigger_pd[trigger_pd.gemroc==0].groupby("run").sum().triggers)
    fig.update_layout(
        xaxis_title="Run number",
        yaxis_title="Triggers",
        height=600
    )

    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')))
    # fig.update_xaxes(range=[1300, 1600])
    # fig.update_yaxes(range=[0, 60])
    return fig

def trigger_vs_subrun_plot(trigger_pd, sel_run):
    trigger_pd=trigger_pd[trigger_pd.run==int(sel_run)]
    fig = px.scatter(x=trigger_pd.subrun, y=trigger_pd.triggers, color=trigger_pd.gemroc.astype(str))
    fig.update_layout(
        xaxis_title="Subrun",
        yaxis_title="Triggers",
        height=600
    )

    # fig.update_traces(marker=dict(size=12,
    #                               line=dict(width=2,
    #                                         color='DarkSlateGrey')))
    # fig.update_xaxes(range=[1300, 1600])
    # fig.update_yaxes(range=[0, 60])
    return fig
if __name__ == '__main__':
    debug = True
    app.run_server(debug=True)
