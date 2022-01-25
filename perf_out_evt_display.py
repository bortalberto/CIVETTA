import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go
from tqdm import tqdm
import os
import numpy as np
# import ROOT as R
# import plotly.io as pio
import time
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit
import scipy.integrate
import ROOT as R
import  sys
import configparser
import perf
from plotly.subplots import make_subplots


def de_correct_process(pos_x, pos_y, corr, planar):
    rev_corr = corr[::-1]
    for correction in corr:
        angle = (correction[f"{int(planar)}_x"][0] - correction[f"{int(planar)}_y"][0]) / 2
        pos_x_0 = pos_x - correction[f"{int(planar)}_y"][1] + np.multiply(angle, (pos_y))
        pos_y_0 = pos_y - correction[f"{int(planar)}_x"][1] - np.multiply(angle, (pos_x))
        pos_x = pos_x_0
        pos_y = pos_y_0
    return pos_x, pos_y

class event_visualizer:
    """
    Class to manage the event visualizer without pre-elaborate the data each time
    """


    def __init__(self, cluster_pd_1D,cluster_pd_match, tracks_pd, hit_pd, eff_pd, put, correction):
        self.cluster_pd_1D  = cluster_pd_1D
        self.cluster_pd_1D["cl_pos_x_cm"] = cluster_pd_1D.cl_pos_x * 0.0650
        self.cluster_pd_1D["cl_pos_y_cm"] = cluster_pd_1D.cl_pos_y * 0.0650
        self.cluster_pd_match = cluster_pd_match
        self.tracks_pd = tracks_pd
        self.put = put
        self.eff_pd = eff_pd
        self.tracks_pd["prev_pos_put_x"] = self.tracks_pd["fit_x"].apply(lambda x: x[0]) * self.put * 10 + tracks_pd["fit_x"].apply(
            lambda x: x[1])
        self.tracks_pd["prev_pos_put_y"] = self.tracks_pd["fit_y"].apply(lambda x: x[0]) * self.put * 10 + tracks_pd["fit_y"].apply(
            lambda x: x[1])
        self.correction = correction
        self.data_pd = hit_pd
        self.event_eff_list = self.eff_pd["count"].unique()



    def plot_evt(self, event):
        """
        Plot the event, including the supposed position and the non efficient clusters
        """
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        #     event=int(np.random.choice(cluster_pd_1D_match[(cluster_pd_1D_match.cl_pos_x_cm.notna()) &(cluster_pd_1D_match.cl_size>5)]["count"].values))
        #     event= 511
        if np.isin(event, self.event_eff_list):
            evt_for_eff = True
            efficient_x = self.eff_pd.loc[self.eff_pd["count"] == event].eff_x.values[0]
        else:
            evt_for_eff = False
            efficient_x = False

        if evt_for_eff and efficient_x:
            this_evt_tracks_pd = self.tracks_pd[self.tracks_pd["count"] == event]
            this_evt_cluster = self.cluster_pd_match[
                (self.cluster_pd_match["count"] == event) & (self.cluster_pd_match.cl_pos_x_cm.notna())]

        if evt_for_eff and not efficient_x:
            this_evt_tracks_pd = self.tracks_pd[self.tracks_pd["count"] == event]
            cluster_pd_1D_c = self.cluster_pd_1D[(self.cluster_pd_1D["count"] == event) & (self.cluster_pd_1D.planar == self.put)]
            clusters = cluster_pd_1D_c.apply(lambda x: perf.apply_correction_eff(x, this_evt_tracks_pd.prev_pos_put_x.values[0], this_evt_tracks_pd.prev_pos_put_y.values[0], self.correction), axis=1)
            clusters["prev_pos_x_cm"] = this_evt_tracks_pd.prev_pos_put_x.values[0]
            clusters["prev_pos_y_cm"] = this_evt_tracks_pd.prev_pos_put_y.values[0]
            this_evt_cluster = clusters.iloc[(clusters['cl_pos_x_cm'] - this_evt_tracks_pd.prev_pos_put_x.values[0]).abs().argsort()[:1]]

        data_pd_evt = self.data_pd[(self.data_pd["count"] == event) & (self.data_pd["planar"] == self.put) & (self.data_pd["strip_x"] > 0) & (
                    self.data_pd["l1ts_min_tcoarse"] > 1370) & (self.data_pd["l1ts_min_tcoarse"] < 1440)]
        if evt_for_eff:
            hit_ids = this_evt_cluster.hit_ids.values[0]
        else:
            hit_ids = [-1]
        data_pd_evt.loc[:,"cluster_eff"] = data_pd_evt["hit_id"].isin(hit_ids)
        color_discrete_map = {False: 'cyan', True: 'red'}
        data_pd_evt.loc[:,"time"] = (1569 - data_pd_evt["l1ts_min_tcoarse"])

        fig = px.scatter(data_pd_evt, "strip_x", "charge_SH", color="cluster_eff",
                         color_discrete_map=color_discrete_map)
        fig2 = px.scatter(data_pd_evt, "strip_x", "time", color_discrete_sequence=["grey"])
        fig.update_traces(yaxis="y2")
        #     fig2.update_layout(name="time")
        if len (fig.data) > 1:
            fig.data[-1].name = 'Nearest cluster'
            fig.data[-2].name = 'Other hits'

        fig2.data[-1].name = 'Time'
        fig2.data[-1].showlegend = True
        subfig.add_traces(fig.data)
        subfig.add_traces(fig2.data)
        subfig.update_xaxes(dict(
            title="Strip X",
            range=[0, 128]))

        subfig.update_yaxes(dict(
            title="Charge_SH [fC]",
            range=[-10, 60],
            side="left"
        ), secondary_y=True)
        subfig.update_yaxes(dict(
            title="Time_wrt_trigger [clk]",
            range=[1569 - 1440, 1569 - 1370],
            side="right"
        ), secondary_y=False)
        if evt_for_eff:
            pos_x = de_correct_process(this_evt_cluster.cl_pos_x_cm.values[0], this_evt_tracks_pd.prev_pos_put_y.values[0],
                                       self.correction, self.put)
            pos_x_prev = de_correct_process(this_evt_tracks_pd.prev_pos_put_x.values[0],
                                            this_evt_tracks_pd.prev_pos_put_y.values[0], self.correction, self.put)

            subfig.add_vline(pos_x_prev[0] / 0.0650, line=dict(color="orange", dash="dot", width=0.7), name="Prev pos")
            subfig.add_vline(pos_x[0] / 0.0650, line=dict(color="red", dash="dot", width=0.7), name="Nearest cl pos")
        if not evt_for_eff:
            evt_type = " No good track"
        else:
            if efficient_x:
                evt_type = " Efficient"
            else:
                evt_type = " Not efficient"
        subfig.update_layout(title=f"Event {event}, planar {self.put}, vista X" + evt_type)
        subfig.update_layout(template="plotly_dark")
        subfig.show()
        #     lap.write_html(subfig, f"eff_evt_30gradi_x_{i}", width=750)

        # Y
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        if np.isin(event, self.event_eff_list):
            evt_for_eff = True
            efficient_y = self.eff_pd.loc[self.eff_pd["count"] == event].eff_y.values[0]
        else:
            evt_for_eff = False
            efficient_y = False

        if evt_for_eff and efficient_y:
            this_evt_tracks_pd = self.tracks_pd[self.tracks_pd["count"] == event]
            this_evt_cluster = self.cluster_pd_match[
                (self.cluster_pd_match["count"] == event) & (self.cluster_pd_match.cl_pos_y_cm.notna())]

        if evt_for_eff and not efficient_y:
            this_evt_tracks_pd = self.tracks_pd[self.tracks_pd["count"] == event]
            cluster_pd_1D_c = self.cluster_pd_1D[(self.cluster_pd_1D["count"] == event) & (self.cluster_pd_1D.planar == self.put)]
            clusters = cluster_pd_1D_c.apply(lambda x: perf.apply_correction_eff(x, this_evt_tracks_pd.prev_pos_put_x.values[0], this_evt_tracks_pd.prev_pos_put_y.values[0], self.correction), axis=1)
            clusters["prev_pos_x_cm"] = this_evt_tracks_pd.prev_pos_put_x.values[0]
            clusters["prev_pos_y_cm"] = this_evt_tracks_pd.prev_pos_put_y.values[0]
            this_evt_cluster = clusters.iloc[(clusters['cl_pos_y_cm'] - this_evt_tracks_pd.prev_pos_put_y.values[0]).abs().argsort()[:1]]

        data_pd_evt = self.data_pd[(self.data_pd["count"] == event) & (self.data_pd["planar"] == self.put) & (self.data_pd["strip_y"] > 0) & (
                self.data_pd["l1ts_min_tcoarse"] > 1370) & (self.data_pd["l1ts_min_tcoarse"] < 1440)]
        if evt_for_eff:
            hit_ids = this_evt_cluster.hit_ids.values[0]
        else:
            hit_ids = [-1]
        data_pd_evt.loc[:, "cluster_eff"] = data_pd_evt["hit_id"].isin(hit_ids)
        color_discrete_map = {False: 'cyan', True: 'red'}
        data_pd_evt.loc[:, "time"] = (1569 - data_pd_evt["l1ts_min_tcoarse"])

        fig = px.scatter(data_pd_evt, "strip_y", "charge_SH", color="cluster_eff",
                         color_discrete_map=color_discrete_map)
        fig2 = px.scatter(data_pd_evt, "strip_y", "time", color_discrete_sequence=["grey"])
        fig.update_traces(yaxis="y2")
        #     fig2.update_layout(name="time")
        if len (fig.data) > 1:
            fig.data[-1].name = 'Nearest cluster'
            fig.data[-2].name = 'Other hits'

        fig2.data[-1].name = 'Time'
        fig2.data[-1].showlegend = True
        subfig.add_traces(fig.data)
        subfig.add_traces(fig2.data)
        subfig.update_xaxes(dict(
            title="Strip Y",
            range=[0, 128]))

        subfig.update_yaxes(dict(
            title="Charge_SH [fC]",
            range=[-10, 60],
            side="left"
        ), secondary_y=True)
        subfig.update_yaxes(dict(
            title="Time_wrt_trigger [clk]",
            range=[1569 - 1440, 1569 - 1370],
            side="right"
        ), secondary_y=False)
        if evt_for_eff:
            pos_x = de_correct_process(this_evt_tracks_pd.prev_pos_put_x.values[0], this_evt_cluster.cl_pos_y_cm.values[0],
                                       self.correction, self.put)
            pos_x_prev = de_correct_process(this_evt_tracks_pd.prev_pos_put_x.values[0],
                                            this_evt_tracks_pd.prev_pos_put_y.values[0], self.correction, self.put)

            subfig.add_vline(pos_x_prev[1] / 0.0650, line=dict(color="orange", dash="dot", width=0.7), name="Prev pos")
            subfig.add_vline(pos_x[1] / 0.0650, line=dict(color="red", dash="dot", width=0.7), name="Nearest cl pos")
        if not evt_for_eff:
            evt_type = " No good track"
        else:
            if efficient_y:
                evt_type = " Efficient"
            else:
                evt_type = " Not efficient"
        subfig.update_layout(title=f"Event {event}, planar {self.put}, vista Y"+ evt_type)
        subfig.update_layout(template="plotly_dark")
        return subfig
#     lap.write_html(subfig, f"eff_evt_30gradi_y_{i}", width=750)
#     lap.write_html(fig, f"eff_evt_{i}_3D", width=750)
