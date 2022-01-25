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
import sys
import configparser
import perf
from plotly.subplots import make_subplots
from scipy.stats import poisson


def de_correct_process(pos_x, pos_y, corr, planar):
    rev_corr = corr[::-1]
    for correction in corr:
        angle = (correction[f"{int(planar)}_x"][0] - correction[f"{int(planar)}_y"][0]) / 2
        pos_x_0 = pos_x - correction[f"{int(planar)}_y"][1] + np.multiply(angle, (pos_y))
        pos_y_0 = pos_y - correction[f"{int(planar)}_x"][1] - np.multiply(angle, (pos_x))
        pos_x = pos_x_0
        pos_y = pos_y_0
    return pos_x, pos_y


def de_correct_process_pd(row, corr):
    planar = row.PUT
    pos_x = row.pos_x
    pos_y = row.pos_y
    rev_corr = corr[::-1]
    for correction in corr:
        angle = (correction[f"{int(planar)}_x"][0] - correction[f"{int(planar)}_y"][0]) / 2
        pos_x = pos_x - correction[f"{int(planar)}_y"][1] + angle * (pos_y)
        pos_y = pos_y - correction[f"{int(planar)}_x"][1] - angle * (pos_x)

    return pos_x, pos_y


class event_visualizer:
    """
    Class to manage the event visualizer without pre-elaborate the data each time
    """

    def __init__(self, cluster_pd_1D, cluster_pd_match, tracks_pd, hit_pd, eff_pd, put, correction):
        self.cluster_pd_1D = cluster_pd_1D
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
        data_pd_evt.loc[:, "cluster_eff"] = data_pd_evt["hit_id"].isin(hit_ids)
        color_discrete_map = {False: 'cyan', True: 'red'}
        data_pd_evt.loc[:, "time"] = (1569 - data_pd_evt["l1ts_min_tcoarse"])

        fig = px.scatter(data_pd_evt, "strip_x", "charge_SH", color="cluster_eff",
                         color_discrete_map=color_discrete_map)
        fig2 = px.scatter(data_pd_evt, "strip_x", "time", color_discrete_sequence=["grey"])
        fig.update_traces(yaxis="y2")
        #     fig2.update_layout(name="time")
        if len(fig.data) > 1:
            fig.data[-1].name = 'Nearest cluster'
            fig.data[-2].name = 'Other hits'
        if data_pd_evt.shape[0] > 0:
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
        subfig_x = subfig
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
        if len(fig.data) > 1:
            fig.data[-1].name = 'Nearest cluster'
            fig.data[-2].name = 'Other hits'
        if data_pd_evt.shape[0] > 0:
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
        subfig.update_layout(title=f"Event {event}, planar {self.put}, vista Y" + evt_type)
        subfig.update_layout(template="plotly_dark")
        subgix_y = subfig
        return (subfig_x, subgix_y)


#     lap.write_html(subfig, f"eff_evt_30gradi_y_{i}", width=750)
#     lap.write_html(fig, f"eff_evt_{i}_3D", width=750)


class eff_calculation:
    """
    Class to calculate the
    """

    def __init__(self, eff_pd, hit_pd, log_path, correction):
        self.eff_pd = eff_pd
        self.hit_pd = hit_pd
        self.log_path = log_path
        self.correction = correction

    def calc_eff(self):
        """
        Calculate the efficiency and the probability to acquire one noise hit instead of signal
        """
        with open(self.log_path, "r") as logfile:
            view = "x"
            tol_x = [float(line.split("+/-")[1].split()[0]) for line in logfile if f"Residual {view}" in line]
        with open(self.log_path, "r") as logfile:
            view = "y"
            tol_y = [float(line.split("+/-")[1].split()[0]) for line in logfile if f"Residual {view}" in line]
        time_win = (1440 - 1370) * 6.25 * 1e-9

        for put in range(0, 4):
            print(f"\n---\nPlanar {put} ")
            eff_pd = self.eff_pd[self.eff_pd.PUT == put]
            eff_pd.loc[:,"pos_x_pl"], eff_pd.loc[:,"pos_y_pl"] = zip(*eff_pd.apply(lambda x: de_correct_process_pd(x, self.correction), axis=1))
            eff_pd_c = eff_pd

            k = eff_pd_c[(eff_pd_c.eff_x) & (eff_pd_c.pos_x_pl > 3) & (eff_pd_c.pos_x_pl < 8) & (eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8)].count().eff_x
            n = eff_pd_c[(eff_pd_c.pos_x_pl > 3) & (eff_pd_c.pos_x_pl < 8) & (eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8)].count().eff_x
            eff_x_good = k / n
            eff_x_good_error = (((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) ** 2) / ((n + 2) ** 2)) ** (1 / 2)
            print(f"X: {eff_x_good:.4f} +/- {eff_x_good_error:.4f}")
            rate_strip_avg = (self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (self.hit_pd.strip_x > 0)].channel.count()) / (self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            error_rate_strip = ((self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (self.hit_pd.strip_x > 0)].channel.count()) ** (1/2) ) / (self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            rate_strip_avg = rate_strip_avg * time_win
            error_rate_strip = error_rate_strip * time_win
            prob_noise_eff = 1 - (poisson.pmf(k=0, mu=rate_strip_avg)) ** round(tol_x[put] * 2 / 0.0650 * 2)
            prob_noise_eff_err = np.exp(rate_strip_avg)*error_rate_strip
            real_eff = (eff_x_good - prob_noise_eff) / (1 - prob_noise_eff)
            error_real_eff = ((eff_x_good_error/(1-prob_noise_eff))**2 + (prob_noise_eff_err/(1-prob_noise_eff**2))**2) ** (1/2)
            print(f"Prob noise eff={prob_noise_eff} +/- {prob_noise_eff_err}")
            print(f"Real eff = {real_eff} +/- {error_real_eff}")
            print(f"---")

            k = eff_pd_c[(eff_pd_c.eff_y) & (eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8) & (eff_pd_c.pos_x_pl > 3) & (eff_pd_c.pos_x_pl < 8)].count().eff_y
            n = eff_pd_c[(eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8) & (eff_pd_c.pos_x_pl > 3) & (eff_pd_c.pos_x_pl < 8)].count().eff_y
            eff_y_good = k / n
            eff_y_good_error = (((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) ** 2) / ((n + 2) ** 2)) ** (1 / 2)
            print(f"Y: {eff_y_good:.4f} +/- {eff_y_good_error:.4f}")
            rate_strip_avg = (self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (self.hit_pd.strip_y > 0)].channel.count()) / (self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            error_rate_strip = ((self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (self.hit_pd.strip_y > 0)].channel.count()) ** (1/2) ) / (self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            rate_strip_avg = rate_strip_avg * time_win
            error_rate_strip = error_rate_strip * time_win
            prob_noise_eff = 1 - (poisson.pmf(k=0, mu=rate_strip_avg)) ** round(tol_y[put] * 2 / 0.0650)
            prob_noise_eff_err = np.exp(rate_strip_avg)*error_rate_strip
            real_eff = (eff_y_good - prob_noise_eff) / (1 - prob_noise_eff)
            error_real_eff = ((eff_y_good_error/(1-prob_noise_eff))**2 + (prob_noise_eff_err/(1-prob_noise_eff**2))**2) ** (1/2)
            print(f"Prob noise eff={prob_noise_eff} +/- {prob_noise_eff_err}")
            print(f"Real eff = {real_eff} +/- {error_real_eff}")
            print(f"---")

        for put in range(0,4):
            #     matching_clusters=pd.read_pickle(os.path.join(eff_path, f"match_cl_{put}.gzip"), compression="gzip")
            print(f"Planar {put} ")
            eff_pd = self.eff_pd[self.eff_pd.PUT == put]
            eff_pd["pos_x_pl"], eff_pd["pos_y_pl"] = zip(*eff_pd.apply(lambda x: de_correct_process_pd(x, self.correction), axis=1))

            eff_pd_c = eff_pd

            k = eff_pd_c[(eff_pd_c.eff_x) & (eff_pd_c.eff_y) & (eff_pd_c.pos_x_pl > 3) & (eff_pd_c.pos_x_pl < 8) & (eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8)].count().eff_x
            n = eff_pd_c[(eff_pd_c.pos_x_pl > 3) & (eff_pd_c.pos_x_pl < 8) & (eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8)].count().eff_x
            eff_x_good = k / n
            eff_x_good_error = (((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) ** 2) / ((n + 2) ** 2)) ** (1 / 2)

            rate_strip_avg = (self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (self.hit_pd.strip_x>0)].channel.count()) / (self.hit_pd["count"].nunique() * (self.hit_pd["l1ts_min_tcoarse"].max() - 1460) * 6.25 * 1e-9) / 123
            rate_strip_avg = rate_strip_avg * time_win
            prob_noise_effx = 1 - (poisson.pmf(k=0, mu=rate_strip_avg)) ** round(tol_x[put] * 2 / 0.0650 * 2)
            rate_strip_avg = (self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (self.hit_pd.strip_y>0)].channel.count()) / (self.hit_pd["count"].nunique() * (self.hit_pd["l1ts_min_tcoarse"].max() - 1460) * 6.25 * 1e-9) / 123
            rate_strip_avg = rate_strip_avg * time_win
            prob_noise_effy = 1 - (poisson.pmf(k=0, mu=rate_strip_avg)) ** round(tol_y[put] * 2 / 0.0650)
            prob_noise_eff = prob_noise_effx + prob_noise_effy - prob_noise_effx*prob_noise_effy
            print(f"2D eff: {eff_x_good:.4f} +/- {eff_x_good_error:.4f}")
            print(prob_noise_eff)
            print(f"Real eff = {(eff_x_good - prob_noise_eff) / (1 - prob_noise_eff)}")


class res_measure:
    """
    Class for the calculation of resolution
    """

    def __init__(self):
        pass
    def calc_res(self):
        return null