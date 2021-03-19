
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
import ROOT as R
import log_loader_time
import os
import json
import glob2

R.gROOT.SetBatch(1)
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
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server,title="Data visual planars")

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
for (dirpath, dirnames, filenames) in walk(data_folder+"/raw_root"):
            for dirname in dirnames:
                if dirname.isdigit():
                    avaible_runs.append(int(dirname))
avaible_runs.sort()
fig_0 = go.Figure()
fig_1 = go.Figure()
fig_2 = go.Figure()
fig_3 = go.Figure()

app.layout = html.Div(children=[
    html.H1(children='Planar setup data visualization '),
    
    html.Div([
    html.H6("Run"),
    dcc.Dropdown(
        id='sel_run',
        options=[{'label': i, 'value': i} for i in avaible_runs],
        value='15'
        )

    ], style={'width': '5%', 'display': 'inline-block'}),
    
    
    html.Div([
    html.H6("Plot type"),
    dcc.Dropdown(
       id='plot_opt',
       options=[
           {"label":"Charge vs time","value":"Charge vs time"},
           {"label":"Signal/noise ratio vs time", "value": "Signal ratio"},
           {"label":"Noise vs time", "value": "Noise"},
           {"label":"X strips","value":"X strips"},
           {"label":"Y strips","value":"Y strips"},
           {"label": "N° Clusers vs time", "value": "Clusters vs time"},
           {"label":"Signal heatmap (2D clusters)","value":"Signal heatmap"},
           {"label":"Distr charge clusters (2D clusters)","value":"Distr charge clusters"}
               ],
       value='Charge vs time'
        ),
    ], style={'width': '15%', 'display': 'inline-block'}),
    
    html.Div([
    html.H6("Time window"),
    dcc.Dropdown(
       id='window_opt',
       options=[
           {"label":"All","value":"All"},
           {"label":f"Signal ","value":"Signal"},
           {"label":"Noise","value":"Noise"}
               ],
       value='All'
        ),
    ], style={'width': '10%', 'display': 'inline-block'}),

    
    html.Div([
    html.H6("Subrun options"),
    dcc.Dropdown(
       id='sel_options',
       options=[{"label":"All","value":"All"},{ "label":"Last 10","value":"Last 10"}, {"label":"Select","value":"Select"}],
       value='All'
        ),
    ], style={'width': '10%', 'display': 'inline-block'}),


    html.Div([
    html.H6("Sel subrun"),
    dcc.Dropdown(
       id='sel_subrun',
       value='0'
        ),
    ], style={'width': '5%', 'display': 'none'},id='sel_dub_div'),

    html.Div([
        html.H6("Cluster selection"),
        dcc.Dropdown(
            id='sel_binning',
            options=[{"label": 0.5, "value": 0.5}, {"label": 1, "value": 1}, {"label": 2, "value": 2}, {"label": 4, "value": 4}],
            value=1
        ),
    ], style={'width': '10%', 'display': 'none'}, id='sel_binning_div'),

    html.Div([
        html.H6("Sel binning (strips)"),
        dcc.Dropdown(
            id='sel_binning',
            options=[{"label": 0.5, "value": 0.5}, {"label": 1, "value": 1}, {"label": 2, "value": 2}, {"label": 4, "value": 4}],
            value=1
        ),
    ], style={'width': '10%', 'display': 'none'}, id='sel_binning_div'),

    html.Br(),
    html.Button('Plot', id='plot_but', n_clicks=0),
    html.Div(id='display_sel_trigger',
             children=''),
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
    )], style={'width': '50%', 'display': 'inline-block'})
    
    ])
    
    

@app.callback(
    Output('data_visual_0', 'figure'),
    Output('data_visual_1', 'figure'),
    Output('data_visual_2', 'figure'),
    Output('data_visual_3', 'figure'),
    Output('display_sel_trigger','children'),
    Input('plot_but', 'n_clicks'),
    State('sel_run', 'value'),
    State('plot_opt', 'value'),
    State('window_opt', 'value'),
    State('sel_options', 'value'),
    State('sel_subrun', 'value'),
    State('sel_binning', 'value'),)


def update_graph(n_clicks, sel_run, plot_opt,window_opt, sel_options,sel_subrun, sel_binning):
    fig_list=[]
    ### Plot da fare sugli HIT
    signal_lower_limit, signal_upper_limit=load_config_signal_limits(sel_run)
    if plot_opt in ("Charge vs time", "X strips", "Y strips","Signal ratio","Noise"):
        data_pd = pd.read_pickle("{}/raw_root/{}/hit_data.pickle.gzip".format(data_folder,sel_run), compression="gzip")
        data_pd_pre_0=data_pd[(data_pd.l1ts_min_tcoarse<1600) & (data_pd.l1ts_min_tcoarse>1300) & (data_pd.charge_SH>0)]
        if window_opt=="Signal":
            data_pd_pre_1=data_pd_pre_0[(data_pd_pre_0.l1ts_min_tcoarse<signal_upper_limit) & (data_pd_pre_0.l1ts_min_tcoarse>signal_lower_limit)]
        elif window_opt=="Noise":
            data_pd_pre_1=data_pd_pre_0[(data_pd_pre_0.l1ts_min_tcoarse>signal_upper_limit) | (data_pd_pre_0.l1ts_min_tcoarse<signal_lower_limit)]
        else :
            data_pd_pre_1=data_pd_pre_0

        if sel_options=="Last 10":
            subruns_list=data_pd_pre_1.subRunNo.unique()
            subruns_list.sort()
            data_pd_pre_2=data_pd_pre_1[data_pd_pre_1.subRunNo.isin(subruns_list[-10:])]
        elif sel_options=="Select":
            data_pd_pre_2=data_pd_pre_1[data_pd_pre_1.subRunNo == sel_subrun]
        else:
            data_pd_pre_2=data_pd_pre_1

        for planar in range (0,4):
            data_pd_cut_2=data_pd_pre_2[ (data_pd_pre_2.planar == planar) ]

            if len (data_pd_cut_2) > 0:

                if plot_opt=="Charge vs time":
                    fig = charge_vs_time_plot(data_pd_cut_2)

                elif plot_opt=="X strips":
                    fig = strips_plot(data_pd_cut_2,"x")

                elif plot_opt=="Y strips":
                    fig = strips_plot(data_pd_cut_2,"y")

                elif plot_opt=="Signal ratio":
                    fig = signal_ratio_plot(data_pd_cut_2,sel_run,planar)

                elif plot_opt=="Noise":
                    fig = noise_hits_plot(data_pd_cut_2, sel_run, planar)
                else:  # Catcher
                    fig = go.Figure()

            else:
                fig=go.Figure()
                fig.update_layout(template=no_data_template)
                
            fig_list.append(fig)


        trig_tot=0
        for sub in data_pd_pre_2.subRunNo.unique():
            trig_tot+=data_pd_pre_2[data_pd_pre_2.subRunNo == sub]["count"].max()
        trigger_string=f"Displayng {trig_tot} triggers"
        fig_list.append(trigger_string)

    ## Plot da fare sui clusters 2D
    if plot_opt in ("Signal heatmap" , "Distr charge clusters", "Clusters vs time"):
        total_clusters = 0
        cluster_pd_2D = pd.read_pickle("{}/raw_root/{}/cluster_pd_2D.pickle.gzip".format(data_folder, sel_run), compression="gzip")
        for planar in range(0, 4):
            cluster_pd_2D_pre_1=cluster_pd_2D[cluster_pd_2D.planar==planar]
            if sel_options == "Last 10":
                subruns_list = cluster_pd_2D.subrun.unique()
                subruns_list.sort()
                cluster_pd_2D_pre_2 = cluster_pd_2D_pre_1[cluster_pd_2D_pre_1.subrun.isin(subruns_list[-10:])]

            elif sel_options == "Select":
                cluster_pd_2D_pre_2 = cluster_pd_2D_pre_1[cluster_pd_2D_pre_1.subrun == sel_subrun]

            else:
                cluster_pd_2D_pre_2 = cluster_pd_2D_pre_1

            if len(cluster_pd_2D_pre_2)>0:
                total_clusters+=len(cluster_pd_2D_pre_2)
                if plot_opt == "Signal heatmap":
                    fig = signal_heatmap_plot(cluster_pd_2D_pre_2)

                elif plot_opt == "Clusters vs time":
                    fig = clusters_vs_tim_plot(cluster_pd_2D_pre_2, sel_run)


                elif plot_opt == "Distr charge clusters":
                    fig = distr_charge_plot_2D(cluster_pd_2D_pre_2)

                else: # Catcher
                    fig = go.Figure()

                fig_list.append(fig)

            else:
                fig = go.Figure()
                fig.update_layout(template=no_data_template)
                fig_list.append(fig)

        # Aggiungi indicazione sul numero di trigger
        data_pd = pd.read_pickle("{}/raw_root/{}/hit_data.pickle.gzip".format(data_folder,sel_run), compression="gzip")
        if sel_options == "Last 10":
            subruns_list = data_pd.subRunNo.unique()
            subruns_list.sort()
            data_pd_pre_2 = data_pd[data_pd.subRunNo.isin(subruns_list[-10:])]
        elif sel_options == "Select":
            data_pd_pre_2 = data_pd[data_pd.subRunNo == sel_subrun]
        else:
            data_pd_pre_2 = data_pd
        trig_tot=0
        for sub in data_pd_pre_2.subRunNo.unique():
            trig_tot+=data_pd_pre_2[data_pd_pre_2.subRunNo == sub]["count"].max()
        fig_list.append(f"{total_clusters} clusters in {trig_tot} triggers")
        
    # Plot da fare su risultato selezione tracciamento
    return fig_list




@app.callback(
    Output('sel_subrun', 'options'),
    Output('sel_run', 'options'),
    Input('sel_run', 'value'),

)
def update_list(sel_run):
    data_pd = pd.read_pickle("{}/raw_root/{}/hit_data.pickle.gzip".format(data_folder,sel_run), compression="gzip")
    available_subs = np.sort(data_pd['subRunNo'].unique())
    avaible_runs=[]
    for (dirpath, dirnames, filenames) in walk(data_folder+"/raw_root"):
                for dirname in dirnames:
                    if dirname.isdigit():
                        avaible_runs.append(int(dirname))
    avaible_runs.sort()
    return [ {'label':name, 'value':name} for name in available_subs],[{'label': i, 'value': i} for i in avaible_runs]



@app.callback(
    Output('sel_dub_div', 'style'),
    Input('sel_options', 'value'),

)
def update_list(sub_option):
    if sub_option=="Select":
        return {'width': '25%', 'display': 'inline-block'}
    else:
        return {'width': '25%', 'display': 'none'}

@app.callback(
    Output('sel_binning_div', 'style'),
    Input('plot_opt', 'value'),

)
def update_list(sub_option):
    if sub_option=="Signal heatmap":
        return {'width': '25%', 'display': 'inline-block'}
    else:
        return {'width': '25%', 'display': 'none'}


def fill_hist_and_norm_and_fit_landau(cluster_pd, norm=1, nbins=600, range_b=300):
    h=R.TH1F(f"h_{cluster_pd.planar.unique()[0]}",f"h_{cluster_pd.planar.unique}", nbins, 0, range_b)
    for x in cluster_pd.cl_charge:
        h.Fill(x)
    # h.Scale(1/norm)
    h.Fit("landau","Q")
    fit_raw=(h.GetFunction("landau").GetParameters())
    fit_err=(h.GetFunction("landau").GetParErrors())

    fit_list=float(fit_raw[0]), float(fit_raw[1]), float(fit_raw[2]),float(fit_err[0]), float(fit_err[1]), float(fit_err[2])
    return fit_list

def calculate_y_landau(fit,x):
    y=R.TMath.Landau(x,fit[1],fit[2])*fit[0]
    return y

def calculater_middle_time(start_dict,end_dict):
    central_time_dict={}
    for subrun in start_dict.keys():
        central_time_dict[subrun]=start_dict[subrun]+ round((end_dict[subrun]-start_dict[subrun])/2)
    return (central_time_dict)

def load_config_signal_limits(run):
    conf_path=data_folder+"/raw_root/"+str(run)+"/analisis_config"
    if os.path.isfile(conf_path):
        with open (conf_path, 'r') as conf_file:
            conf_dict = json.load(conf_file)
            if conf_dict["time_window"]:
                    return conf_dict["time_window"][0],conf_dict["time_window"][1]
    signal_upper_limit = config["GLOBAL"].getint("signal_window_upper_limit")
    signal_lower_limit = config["GLOBAL"].getint("signal_window_lower_limit")
    return signal_lower_limit, signal_upper_limit


def extract_num_triggers_subrun(filename):
    with open (filename,'r') as log_file:
        for line in log_file.readlines():
            if "total packets" in line:
                return line.split(" ")[-1]

    return 0
def extract_num_triggers_run(data_raw_folder, run_number):
    return_dict={}
    for filename, subrun in glob2.iglob("{}/RUN_{}/ACQ_log_*".format(data_raw_folder,run_number),with_matches=True):
        return_dict[subrun[0]]=int(extract_num_triggers_subrun(filename))
    return return_dict

def calc_rate_per_sub(row, trig_dict, width):
    sub = row.subRunNo
    if trig_dict[str(sub)] > 0:
        return row["noise_tot"] / (trig_dict[str(sub)]*width)
    else:
        return np.NAN


## Plot functions
def charge_vs_time_plot(data_pd_cut_2):
    fig = px.density_heatmap(data_pd_cut_2, x="l1ts_min_tcoarse", y="charge_SH",
                             title="Charge vs time",
                             marginal_x="histogram",
                             marginal_y="histogram",
                             color_continuous_scale=colorscale,
                             nbinsx=int(data_pd_cut_2.l1ts_min_tcoarse.max() - (data_pd_cut_2.l1ts_min_tcoarse.min())),
                             nbinsy=120)
    fig.update_layout(
        xaxis_title="Trigger time stamp - hit TCoarse ",
        yaxis_title="Charge [fC]",
        height=800
    )
    fig.update_xaxes(range=[1300, 1600])
    fig.update_yaxes(range=[0, 60])
    return fig

def strips_plot(data_pd_cut_2, view):
    data_pd_cut_3 = data_pd_cut_2[data_pd_cut_2[f"strip_{view}"] > 0]
    fig = px.density_heatmap(data_pd_cut_3, x=f"strip_{view}", y="charge_SH",
                             title=f"Charge vs strip {view}",
                             marginal_x="histogram",
                             color_continuous_scale=colorscale,
                             hover_data={'channel': True,  # remove species from hover data
                                         f'strip_{view}': True,  # customize hover for column of y attribute
                                         'gemroc' : True  # add other column, default formatting
                                         },
                             nbinsx=128,
                             range_x=[0, 128],
                             nbinsy=120,
                             range_y=[0, 60])
    fig.update_layout(
        yaxis_title="Charge [fC] ",
        xaxis_title=f"{view} strip",
        height=800
    )
    return fig

def signal_ratio_plot(data_pd_cut_2, sel_run, planar):
    ## Estract time informmation
    time = {}
    time_reader = log_loader_time.reader(data_folder + "/raw_dat/", data_folder + "/time/")
    time_dict=time_reader.elab_on_run_dict(sel_run)
    start, end = time_dict
    time[int(sel_run)] = calculater_middle_time(start, end)
    ## Add time information to PD
    data_pd_cut_2.insert(0, "time_r", 0)
    data_pd_cut_2["time_r"] = data_pd_cut_2.apply(lambda row: time[(int(row.runNo))][str(int(row.subRunNo))], axis=1)
    data_pd_cut_2['time_r'] = pd.to_datetime(data_pd_cut_2['time_r'], unit='s')
    signal_list = []

    ## Create signal and noise count PD
    sin_cut = data_pd_cut_2[((data_pd_cut_2.l1ts_min_tcoarse < signal_upper_limit) & (data_pd_cut_2.l1ts_min_tcoarse > signal_lower_limit) & (data_pd_cut_2.planar == planar) & (data_pd_cut_2.charge_SH > 5))]
    nois_cut = data_pd_cut_2[((data_pd_cut_2.l1ts_min_tcoarse > signal_upper_limit) | (data_pd_cut_2.l1ts_min_tcoarse < signal_lower_limit) & (data_pd_cut_2.planar == planar))]

    sin = sin_cut.groupby(sin_cut.time_r.dt.hour).agg({"charge_SH": "count", "time_r": "last"})
    nois = nois_cut.groupby(nois_cut.time_r.dt.hour).agg({"charge_SH": "count", "time_r": "last"})

    signal_list.append(nois.rename(columns={"charge_SH": "noise_planar_{}".format(planar)}))
    signal_list.append(sin.rename(columns={"charge_SH": "signal_planar_{}".format(planar)}))
    result = pd.concat(signal_list, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    x_data=[x for x in result.time_r]
    y_data=[y for y in result[f"signal_planar_{planar}"]/result[f"noise_planar_{planar}"]]
    fig = px.scatter(
            x=x_data,
            y=y_data
        )

    fig.update_layout(
        yaxis_title="Signal/noise  ratio",
        xaxis_title="Time",
        height=800
    )
    if max(y_data)<1:
        fig.update_yaxes(range=[0, 1])
    return fig


def noise_hits_plot(data_pd_cut_2, sel_run, planar):
    ## Estract time informmation
    time = {}
    time_reader = log_loader_time.reader(data_folder + "/raw_dat/", data_folder + "/time/")
    time_dict = time_reader.elab_on_run_dict(sel_run)
    start, end = time_dict
    time[int(sel_run)] = calculater_middle_time(start, end)
    ## Add time information to PD
    data_pd_cut_2.insert(0, "time_r", 0)
    data_pd_cut_2["time_r"] = data_pd_cut_2.apply(lambda row: time[(int(row.runNo))][str(int(row.subRunNo))], axis=1)
    data_pd_cut_2['time_r'] = pd.to_datetime(data_pd_cut_2['time_r'], unit='s')

    ## Create noise count PD
    nois_cut = data_pd_cut_2[((data_pd_cut_2.l1ts_min_tcoarse > signal_upper_limit) | (data_pd_cut_2.l1ts_min_tcoarse < signal_lower_limit) & (data_pd_cut_2.planar == planar))]
    noise_pd = nois_cut.groupby(nois_cut.subRunNo).agg({"charge_SH": "count", "time_r": "last", "subRunNo": "last"}, )
    noise_pd = noise_pd.rename(columns={"charge_SH": "noise_tot"})
    trig_dict = extract_num_triggers_run(data_folder + "/raw_dat/", sel_run)
    noise_win_width = (1567 - signal_upper_limit + signal_lower_limit - 1299) * 6.25 * 10 ** (-9)
    noise_pd["noise_rate"] = noise_pd.apply(calc_rate_per_sub, args=(trig_dict, noise_win_width), axis=1)
    # noise_pd = noise_pd.groupby(nois_cut.time_r.dt.hour).agg({"noise_rate": "mean", "time_r": "last","subRunNo":"last"}, )
    noise_pd["time_r"] = (noise_pd["time_r"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    noise_pd = noise_pd.groupby(noise_pd.subRunNo.divide(5).round().multiply(5)).agg({"subRunNo": "mean", "time_r": "mean", "noise_rate": "mean"})
    noise_pd["time_r"] = pd.to_datetime(noise_pd["time_r"], unit='s')
    fig = px.scatter(
        noise_pd,
        x="time_r",
        y="noise_rate",
        hover_data=["subRunNo"]
    )

    fig.update_layout(
        yaxis_title="Noise planar [Hz]",
        xaxis_title="Time",
        height=800
    )

    if max(noise_pd.noise_rate) < 5000000:
        fig.update_yaxes(range=[0, 5000000])
    return fig

## Cluster plots

def signal_heatmap_plot(cluster_pd_2D_pre_2):
    fig = px.density_heatmap(x=cluster_pd_2D_pre_2.cl_pos_x, y=cluster_pd_2D_pre_2.cl_pos_y, marginal_x="histogram", marginal_y="histogram", nbinsx=int(128 / sel_binning), nbinsy=int(128 / sel_binning))
    fig.update_layout(
        yaxis_title="Y strips ",
        xaxis_title="X strips",
        height=800
    )
    return fig

def clusters_vs_tim_plot(cluster_pd_2D_pre_2, sel_run):
    time = {}
    time_reader = log_loader_time.reader(data_folder + "/raw_dat/", data_folder + "/time/")
    time_dict = time_reader.elab_on_run_dict(sel_run)
    start, end = time_dict
    time[int(sel_run)] = calculater_middle_time(start, end)
    ## Add time information to PD
    cluster_pd_2D_pre_2.insert(0, "time_r", 0)
    cluster_pd_2D_pre_2["time_r"] = cluster_pd_2D_pre_2.apply(lambda row: time[(int(row.run))][str(int(row.subrun))], axis=1)
    cluster_pd_2D_pre_2['time_r'] = pd.to_datetime(cluster_pd_2D_pre_2['time_r'], unit='s')
    clusters_list = []
    trigger_list = []
    time_list = []
    trig_dict = extract_num_triggers_run(data_folder + "/raw_dat/", sel_run)
    for subrun in cluster_pd_2D_pre_2.subrun.unique():
        trigger_list.append(trig_dict[str(int(subrun))])
        cl_pd = cluster_pd_2D_pre_2[cluster_pd_2D_pre_2.subrun == subrun]
        clusters_list.append(cl_pd.cl_pos_x.count())
        time_list.append(cl_pd.time_r.min())
    fig = px.scatter(x=time_list, y=np.divide(clusters_list, trigger_list), marginal_y="histogram")
    fig.update_layout(
        yaxis_title="Clusters / trigger",
        xaxis_title="Time",
        height=800
    )
    return fig
def distr_charge_plot_2D(cluster_pd_2D_pre_2):
    range_b = 500
    nbins = 500
    cluster_pd_2D_cut = cluster_pd_2D_pre_2[cluster_pd_2D_pre_2.cl_charge < 500]
    fit = fill_hist_and_norm_and_fit_landau(cluster_pd_2D_cut, 1, nbins, range_b)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=cluster_pd_2D_cut.cl_charge, opacity=0.75, xbins=dict(size=range_b / nbins, start=0), name="Charge histogram"))
    fig.update_layout(
        yaxis_title="Count",
        xaxis_title="Cluster charge [fC]",
        height=800
    )
    x_list = np.arange(0, range_b, range_b / nbins)
    y_list = [calculate_y_landau(fit, x) for x in x_list]
    fig.add_trace(go.Scatter(x=x_list, y=y_list, name="Landau fit"))
    textfont = dict(
        family="sans serif",
        size=18,
        color="LightSeaGreen"
    )
    fig.add_annotation(x=0.75, xref="paper", y=0.95, yref="paper",
                       text=f"""AVG Charge: {cluster_pd_2D_cut.cl_charge.mean():.2f} +/- {
                       np.std(cluster_pd_2D_cut.cl_charge) /
                       (cluster_pd_2D_cut.cl_charge.count()) ** (1 / 2):.2f}""", showarrow=False, font=textfont)
    fig.add_annotation(x=0.75, xref="paper", y=0.90, yref="paper",
                       text=f"MPV CHarge: {fit[1]:.2f} +/- {fit[4]:.2f}", showarrow=False, font=textfont)
    fig.add_annotation(x=0.75, xref="paper", y=0.85, yref="paper",
                       text=f"""AVG size(x+y): {cluster_pd_2D_cut.cl_size_tot.mean():.2f} +/- {
                       np.std(cluster_pd_2D_cut.cl_size_tot) /
                       (cluster_pd_2D_cut.cl_size_tot.count()) ** (1 / 2):.2f}""", showarrow=False, font=textfont)

    return fig
if __name__ == '__main__':
    debug = True
    app.run_server(debug=True)
