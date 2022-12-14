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
import root_fit_lib as r_fit
import sympy as sym

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
        self.tracks_pd["prev_pos_put_x"] = self.tracks_pd["fit_x"].apply(lambda x: x[0]) * self.put * 10 + tracks_pd[
            "fit_x"].apply(
            lambda x: x[1])
        self.tracks_pd["prev_pos_put_y"] = self.tracks_pd["fit_y"].apply(lambda x: x[0]) * self.put * 10 + tracks_pd[
            "fit_y"].apply(
            lambda x: x[1])
        self.correction = correction
        self.hit_pd = hit_pd
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
            cluster_pd_1D_c = self.cluster_pd_1D[
                (self.cluster_pd_1D["count"] == event) & (self.cluster_pd_1D.planar == self.put)]
            clusters = cluster_pd_1D_c.apply(
                lambda x: perf.apply_correction_eff(x, this_evt_tracks_pd.prev_pos_put_x.values[0],
                                                    this_evt_tracks_pd.prev_pos_put_y.values[0], self.correction),
                axis=1)
            clusters["prev_pos_x_cm"] = this_evt_tracks_pd.prev_pos_put_x.values[0]
            clusters["prev_pos_y_cm"] = this_evt_tracks_pd.prev_pos_put_y.values[0]
            this_evt_cluster = clusters.iloc[
                (clusters['cl_pos_x_cm'] - this_evt_tracks_pd.prev_pos_put_x.values[0]).abs().argsort()[:1]]

        hit_pd_evt = self.hit_pd[
            (self.hit_pd["count"] == event) & (self.hit_pd["planar"] == self.put) & (self.hit_pd["strip_x"] > 0) & (
                    self.hit_pd["l1ts_min_tcoarse"] > 1370) & (self.hit_pd["l1ts_min_tcoarse"] < 1480)]
        if evt_for_eff :
            if this_evt_cluster.shape[0]>0:
                hit_ids = this_evt_cluster.hit_ids.values[0]
            else:
                hit_ids = [-1]
        else:
            hit_ids = [-1]
        hit_pd_evt.loc[:, "cluster_eff"] = hit_pd_evt["hit_id"].isin(hit_ids)
        color_discrete_map = {False: 'cyan', True: 'red'}
        hit_pd_evt.loc[:, "time"] = (1569 - hit_pd_evt["l1ts_min_tcoarse"])

        fig = px.scatter(hit_pd_evt, "strip_x", "charge_SH", color="cluster_eff",
                         color_discrete_map=color_discrete_map)
        fig2 = px.scatter(hit_pd_evt, "strip_x", "time", color_discrete_sequence=["grey"])
        fig.update_traces(yaxis="y2")
        #     fig2.update_layout(name="time")
        if len(fig.data) > 1:
            fig.data[-1].name = 'Nearest cluster'
            fig.data[-2].name = 'Other hits'
        if hit_pd_evt.shape[0] > 0:
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
            pos_x = de_correct_process(this_evt_cluster.cl_pos_x_cm.values[0],
                                       this_evt_tracks_pd.prev_pos_put_y.values[0],
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
            cluster_pd_1D_c = self.cluster_pd_1D[
                (self.cluster_pd_1D["count"] == event) & (self.cluster_pd_1D.planar == self.put)]
            clusters = cluster_pd_1D_c.apply(
                lambda x: perf.apply_correction_eff(x, this_evt_tracks_pd.prev_pos_put_x.values[0],
                                                    this_evt_tracks_pd.prev_pos_put_y.values[0], self.correction),
                axis=1)
            clusters["prev_pos_x_cm"] = this_evt_tracks_pd.prev_pos_put_x.values[0]
            clusters["prev_pos_y_cm"] = this_evt_tracks_pd.prev_pos_put_y.values[0]
            this_evt_cluster = clusters.iloc[
                (clusters['cl_pos_y_cm'] - this_evt_tracks_pd.prev_pos_put_y.values[0]).abs().argsort()[:1]]

        hit_pd_evt = self.hit_pd[
            (self.hit_pd["count"] == event) & (self.hit_pd["planar"] == self.put) & (self.hit_pd["strip_y"] > 0) & (
                    self.hit_pd["l1ts_min_tcoarse"] > 1370) & (self.hit_pd["l1ts_min_tcoarse"] < 1440)]
        if evt_for_eff:
            hit_ids = this_evt_cluster.hit_ids.values[0]
        else:
            hit_ids = [-1]
        hit_pd_evt.loc[:, "cluster_eff"] = hit_pd_evt["hit_id"].isin(hit_ids)
        color_discrete_map = {False: 'cyan', True: 'red'}
        hit_pd_evt.loc[:, "time"] = (1569 - hit_pd_evt["l1ts_min_tcoarse"])

        fig = px.scatter(hit_pd_evt, "strip_y", "charge_SH", color="cluster_eff",
                         color_discrete_map=color_discrete_map)
        fig2 = px.scatter(hit_pd_evt, "strip_y", "time", color_discrete_sequence=["grey"])
        fig.update_traces(yaxis="y2")
        #     fig2.update_layout(name="time")
        if len(fig.data) > 1:
            fig.data[-1].name = 'Nearest cluster'
            fig.data[-2].name = 'Other hits'
        if hit_pd_evt.shape[0] > 0:
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
            pos_x = de_correct_process(this_evt_tracks_pd.prev_pos_put_x.values[0],
                                       this_evt_cluster.cl_pos_y_cm.values[0],
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

    def __init__(self, eff_pd_joint, hit_pd, log_path, correction ,outpath):
        self.eff_pd = eff_pd_joint
        self.hit_pd = hit_pd
        self.log_path = log_path
        self.correction = correction
        self.outpath = outpath

    def calc_eff(self, planar_list):
        """
        Calculate the efficiency and the probability to acquire one noise hit instead of signal
        """
        with open(self.log_path, "r") as logfile:
            view = "x"
            tol_x = [float(line.split("+/-")[1].split()[0]) for line in logfile if f"Residual {view}" in line]
        with open(self.log_path, "r") as logfile:
            view = "y"
            tol_y = [float(line.split("+/-")[1].split()[0]) for line in logfile if f"Residual {view}" in line]
        with open(self.log_path, "r") as logfile:
            key = [int(line.split(" ")[-1]) for line in logfile if f"Measuring performances on planar" in line]
        tol_x=dict(zip(key, tol_x))
        tol_y=dict(zip(key, tol_y))
        time_win = (1440 - 1370) * 6.25 * 1e-9
        logger = perf.log_writer(self.outpath, 0, "efficiency.txt")
        for put in planar_list:
            print(f"\n---\nPlanar {put} ")
            logger.write_log(f"\n---\nPlanar {put} ")
            eff_pd = self.eff_pd[self.eff_pd.PUT == put]
            eff_pd.loc[:, "pos_x_pl"], eff_pd.loc[:, "pos_y_pl"] = zip(
                *eff_pd.apply(lambda x: de_correct_process_pd(x, self.correction), axis=1))
            eff_pd_c = eff_pd

            k = eff_pd_c[
                (eff_pd_c.eff_x) & (eff_pd_c.pos_x_pl > 3) & (eff_pd_c.pos_x_pl < 8) & (eff_pd_c.pos_y_pl > 3) & (
                        eff_pd_c.pos_y_pl < 8)].count().eff_x
            n = eff_pd_c[(eff_pd_c.pos_x_pl > 3) & (eff_pd_c.pos_x_pl < 8) & (eff_pd_c.pos_y_pl > 3) & (
                    eff_pd_c.pos_y_pl < 8)].count().eff_x
            eff_x_good = k / n
            eff_x_good_error = (((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) ** 2) / ((n + 2) ** 2)) ** (1 / 2)
            print(f"X: {eff_x_good:.4f} +/- {eff_x_good_error:.4f}")
            logger.write_log(f"X: {eff_x_good:.4f} +/- {eff_x_good_error:.4f}")
            rate_strip_avg = (self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_x > -1)].channel.count()) / (
                                     self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            error_rate_strip = ((self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_x > -1)].channel.count()) ** (1 / 2)) / (
                                       self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            rate_strip_avg = rate_strip_avg * time_win
            error_rate_strip = error_rate_strip * time_win
            prob_noise_eff = 1 - (poisson.pmf(k=0, mu=rate_strip_avg)) ** round(tol_x[put] * 2 / 0.0650 * 2)
            prob_noise_eff_err = np.exp(rate_strip_avg) * error_rate_strip
            real_eff = (eff_x_good - prob_noise_eff) / (1 - prob_noise_eff)
            error_real_eff = ((eff_x_good_error / (1 - prob_noise_eff)) ** 2 + (
                    prob_noise_eff_err / (1 - prob_noise_eff ** 2)) ** 2) ** (1 / 2)
            print(f"Prob noise eff = {prob_noise_eff:.3E} +/- {prob_noise_eff_err:.3E}")
            logger.write_log(f"Prob noise eff = {prob_noise_eff:.3E} +/- {prob_noise_eff_err:.3E}")

            print(f"Real eff = {real_eff:.4f} +/- {error_real_eff:.4f}")
            logger.write_log(f"Real eff = {real_eff:.4f} +/- {error_real_eff:.4f}")

            print(f"---")
            logger.write_log(f"---")


            k = eff_pd_c[
                (eff_pd_c.eff_y) & (eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8) & (eff_pd_c.pos_x_pl > 3) & (
                        eff_pd_c.pos_x_pl < 8)].count().eff_y
            n = eff_pd_c[(eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8) & (eff_pd_c.pos_x_pl > 3) & (
                    eff_pd_c.pos_x_pl < 8)].count().eff_y
            eff_y_good = k / n
            eff_y_good_error = (((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) ** 2) / ((n + 2) ** 2)) ** (1 / 2)
            print(f"Y: {eff_y_good:.4f} +/- {eff_y_good_error:.4f}")
            logger.write_log(f"Y: {eff_y_good:.4f} +/- {eff_y_good_error:.4f}")

            rate_strip_avg = (self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_y > -1)].channel.count()) / (
                                     self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            error_rate_strip = ((self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_y > -1)].channel.count()) ** (1 / 2)) / (
                                       self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            rate_strip_avg = rate_strip_avg * time_win
            error_rate_strip = error_rate_strip * time_win
            prob_noise_eff = 1 - (poisson.pmf(k=0, mu=rate_strip_avg)) ** round(tol_y[put] * 2 / 0.0650)
            prob_noise_eff_err = np.exp(rate_strip_avg) * error_rate_strip
            real_eff = (eff_y_good - prob_noise_eff) / (1 - prob_noise_eff)
            error_real_eff = ((eff_y_good_error / (1 - prob_noise_eff)) ** 2 + (
                    prob_noise_eff_err / (1 - prob_noise_eff ** 2)) ** 2) ** (1 / 2)
            print(f"Prob noise eff = {prob_noise_eff:.3E} +/- {prob_noise_eff_err:.3E}")
            logger.write_log(f"Prob noise eff = {prob_noise_eff:.3E} +/- {prob_noise_eff_err:.3E}")
            print(f"Real eff = {real_eff:.4f} +/- {error_real_eff:.4f}")
            logger.write_log(f"Real eff = {real_eff:.4f} +/- {error_real_eff:.4f}")
            print(f"---")
            logger.write_log(f"---")

            print(f"---")
            print(f"AND eff")
            logger.write_log(f"AND eff")
            k = eff_pd_c[
                (eff_pd_c.eff_y) &  (eff_pd_c.eff_x) & (eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8) & (eff_pd_c.pos_x_pl > 3) & (
                        eff_pd_c.pos_x_pl < 8)].count().eff_y
            n = eff_pd_c[(eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8) & (eff_pd_c.pos_x_pl > 3) & (
                    eff_pd_c.pos_x_pl < 8)].count().eff_y
            eff_y_good = k / n
            eff_y_good_error = (((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) ** 2) / ((n + 2) ** 2)) ** (1 / 2)
            print(f"AND: {eff_y_good:.4f} +/- {eff_y_good_error:.4f}")
            logger.write_log(f"AND: {eff_y_good:.4f} +/- {eff_y_good_error:.4f}")
            ## Y
            rate_strip_avg_y = (self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_y > -1)].channel.count()) / (
                                     self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            error_rate_strip_y = ((self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_y > -1)].channel.count()) ** (1 / 2)) / (
                                       self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            rate_strip_avg_y = rate_strip_avg_y * time_win
            error_rate_strip_y = error_rate_strip_y * time_win
            prob_noise_eff_y = 1 - (poisson.pmf(k=0, mu=rate_strip_avg_y)) ** round(tol_y[put] * 2 / 0.0650)
            prob_noise_eff_err_y = np.exp(rate_strip_avg_y) * error_rate_strip_y
            ## X
            rate_strip_avg_x = (self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_x > -1)].channel.count()) / (
                                     self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            error_rate_strip_x = ((self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_x > -1)].channel.count()) ** (1 / 2)) / (
                                       self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            rate_strip_avg_x = rate_strip_avg_x * time_win
            error_rate_strip_x = error_rate_strip_x * time_win
            prob_noise_eff_x = 1 - (poisson.pmf(k=0, mu=rate_strip_avg_x)) ** round(tol_x[put] * 2 / 0.0650)
            prob_noise_eff_err_x = np.exp(rate_strip_avg_x) * error_rate_strip_x

            prob_noise_eff = prob_noise_eff_x * prob_noise_eff_y
            prob_noise_eff_err = ((prob_noise_eff_x * prob_noise_eff_err_y)**2 + (prob_noise_eff_y * prob_noise_eff_err_x)**2) ** (1/2)

            real_eff = (eff_y_good - prob_noise_eff) / (1 - prob_noise_eff)
            error_real_eff = ((eff_y_good_error / (1 - prob_noise_eff)) ** 2 + (
                    prob_noise_eff_err / (1 - prob_noise_eff ** 2)) ** 2) ** (1 / 2)
            print(f"Prob noise eff = {prob_noise_eff:.3E} +/- {prob_noise_eff_err:.3E}")
            logger.write_log(f"Prob noise eff = {prob_noise_eff:.3E} +/- {prob_noise_eff_err:.3E}")
            print(f"Real eff = {real_eff:.4f} +/- {error_real_eff:.4f}")
            logger.write_log(f"Real eff = {real_eff:.4f} +/- {error_real_eff:.4f}")
            print(f"---")
            logger.write_log(f"---")


            print(f"---")

        for put in planar_list:
            #     matching_clusters=pd.read_pickle(os.path.join(eff_path, f"match_cl_{put}.gzip"), compression="gzip")
            print(f"Planar {put} ")
            eff_pd = self.eff_pd[self.eff_pd.PUT == put]
            eff_pd.loc[:, "pos_x_pl"], eff_pd.loc[:, "pos_y_pl"] = zip(
                *eff_pd.apply(lambda x: de_correct_process_pd(x, self.correction), axis=1))

            eff_pd_c = eff_pd

            k = eff_pd_c[(eff_pd_c.eff_x) & (eff_pd_c.eff_y) & (eff_pd_c.pos_x_pl > 3) & (eff_pd_c.pos_x_pl < 8) & (
                    eff_pd_c.pos_y_pl > 3) & (eff_pd_c.pos_y_pl < 8)].count().eff_x
            n = eff_pd_c[(eff_pd_c.pos_x_pl > 3) & (eff_pd_c.pos_x_pl < 8) & (eff_pd_c.pos_y_pl > 3) & (
                    eff_pd_c.pos_y_pl < 8)].count().eff_x
            eff_and_good = k / n
            eff_and_good_error = (((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) ** 2) / ((n + 2) ** 2)) ** (
                    1 / 2)

            rate_strip_avg = (self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_x > 0)].channel.count()) / (self.hit_pd["count"].nunique() * (
                    self.hit_pd["l1ts_min_tcoarse"].max() - 1460) * 6.25 * 1e-9) / 123
            error_rate_strip = ((self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_x > 0)].channel.count()) ** (1 / 2)) / (
                                       self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            rate_strip_avg = rate_strip_avg * time_win
            error_rate_strip = error_rate_strip * time_win
            prob_noise_effx = 1 - (poisson.pmf(k=0, mu=rate_strip_avg)) ** round(tol_x[put] * 2 / 0.0650 * 2)
            prob_noise_eff_errx = np.exp(rate_strip_avg) * error_rate_strip

            rate_strip_avg = (self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_y > 0)].channel.count()) / (self.hit_pd["count"].nunique() * (
                    self.hit_pd["l1ts_min_tcoarse"].max() - 1460) * 6.25 * 1e-9) / 123
            error_rate_strip = ((self.hit_pd[(self.hit_pd.l1ts_min_tcoarse > 1460) & (self.hit_pd.planar == put) & (
                    self.hit_pd.strip_y > 0)].channel.count()) ** (1 / 2)) / (
                                       self.hit_pd["count"].nunique() * (1569 - 1460) * 6.25 * 1e-9) / 123
            rate_strip_avg = rate_strip_avg * time_win
            error_rate_strip = error_rate_strip * time_win
            prob_noise_effy = 1 - (poisson.pmf(k=0, mu=rate_strip_avg)) ** round(tol_y[put] * 2 / 0.0650)
            prob_noise_eff_erry = np.exp(rate_strip_avg) * error_rate_strip

            prob_noise_eff = prob_noise_effx + prob_noise_effy - prob_noise_effx * prob_noise_effy
            prob_noise_eff_err = (((1 - prob_noise_effy) * prob_noise_eff_errx) ** 2 + (
                    (1 - prob_noise_effx) * prob_noise_eff_erry) ** 2) ** (1 / 2)

            real_eff = (eff_and_good - prob_noise_eff) / (1 - prob_noise_eff)
            error_real_eff = ((eff_and_good_error / (1 - prob_noise_eff)) ** 2 + (
                    prob_noise_eff_err / (1 - prob_noise_eff ** 2)) ** 2) ** (1 / 2)

            print(f"2D eff = {eff_and_good:.4f} +/- {eff_and_good_error:.4f}")
            print(f"Real eff = {real_eff:.4f} +/- {error_real_eff:.4f}")
            print(f"---")


class res_measure:
    """
    Class for the calculation of resolution
    """

    def __init__(self, cl_pd, tracks_pd, eff_pd, planar_list):
        """
        :param cl_pd: match pd
        :param tracks_pd:
        :param eff_pd:
        :param put:
        """
        self.cl_pds = {}
        # print (tracks_pd.keys)
        for put in planar_list:
            cl_pd_x, cl_pd_y = self.generate_cl_res_pd(eff_pd, tracks_pd, cl_pd, put)
            self.cl_pds[f"{put}x"] = cl_pd_x
            self.cl_pds[f"{put}y"] = cl_pd_y

    def generate_cl_res_pd(self, eff_pd, tracks_pd, cl_pd, put):
        """
        Generates the 2 pd to fit resisualds
        """
        eff_pd_c = eff_pd[
            (eff_pd.eff_x) & (eff_pd.eff_y) & (eff_pd.PUT == put)]  # Select efficient events in the good region
        good_evt = eff_pd_c["count"].unique()
        tracks_pd = tracks_pd[put]
        cl_pd = cl_pd[put]

        tracks_pd = tracks_pd[tracks_pd["count"].isin(good_evt)]  # Cut the row relative to other events
        cl_pd = cl_pd[(cl_pd["count"].isin(good_evt))]

        tracks_pd.loc[:, "prev_pos_put_x"] = tracks_pd["fit_x"].apply(lambda x: x[0]) * put * 10 + tracks_pd["fit_x"].apply(
            lambda x: x[1])
        tracks_pd.loc[:, "prev_pos_put_y"] = tracks_pd["fit_y"].apply(lambda x: x[0]) * put * 10 + tracks_pd["fit_y"].apply(
            lambda x: x[1])  # Calculate supposed position

        duplicated_cl_event = cl_pd["count"].unique()[cl_pd.groupby("count").agg("size") > 2]  # Drop events with 2 efficient clusters
        cl_pd = cl_pd[~cl_pd["count"].isin(duplicated_cl_event)] # Possiamo fare diveramente?

        tracks_pd = tracks_pd[~tracks_pd["count"].isin(duplicated_cl_event)]
        tracks_pd = tracks_pd.sort_values("count").reset_index(drop=True)

        cl_pd_x = cl_pd.loc[cl_pd.cl_pos_x.notna()].reset_index(drop=True)  # Divides the datatest between x and y and sort them
        cl_pd_y = cl_pd.loc[cl_pd.cl_pos_y.notna()].reset_index(drop=True)
        cl_pd_x = cl_pd_x.sort_values("count").reset_index(drop=True)
        cl_pd_y = cl_pd_y.sort_values("count").reset_index(drop=True)

        cl_pd_x.loc[:, "prev_pos_x_cm"] = tracks_pd.prev_pos_put_x
        cl_pd_y.loc[:, "prev_pos_y_cm"] = tracks_pd.prev_pos_put_y

        cl_pd_x.loc[:, "angle_trk_x"] = tracks_pd["fit_x"].apply(lambda x: x[0])
        cl_pd_y.loc[:, "angle_trk_y"] = tracks_pd["fit_y"].apply(lambda x: x[0])

        cl_pd_x.loc[:, "res_x"] = tracks_pd.prev_pos_put_x - cl_pd_x.loc[:, "cl_pos_x_cm"]
        cl_pd_y.loc[:, "res_y"] = tracks_pd.prev_pos_put_y - cl_pd_y.loc[:, "cl_pos_y_cm"]

        cl_pd_x.loc[:, "cov"] = tracks_pd.cov_x
        cl_pd_y.loc[:, "cov"] = tracks_pd.cov_y

        cl_pd_x["error_tracking"] = cl_pd_x.apply(
            lambda x: x["cov"][1][1] + (x.cl_pos_z_cm ** 2) * x["cov"][0, 0] + x.cl_pos_z_cm * x["cov"][0, 1] + x.cl_pos_z_cm *
                      x["cov"][1, 0], 1)

        cl_pd_y["error_tracking"] = cl_pd_y.apply(
            lambda x: x["cov"][1][1] + (x.cl_pos_z_cm ** 2) * x["cov"][0, 0] + x.cl_pos_z_cm * x["cov"][0, 1] + x.cl_pos_z_cm *
                      x["cov"][1, 0], 1)

        return cl_pd_x, cl_pd_y


    def calc_res(self, planar, view):
        cl_pd = self.cl_pds[f"{planar}{view}"]
        popt_list, pcov_list, res_list, R_list, chi_list, deg_list = r_fit.double_gaus_fit_root(
            pd.DataFrame(cl_pd[f"res_{view}"].apply(lambda x: [x, x, x, x], 1)), view=view)
        return popt_list[0], pcov_list[0], res_list[0], R_list[0], chi_list[0], deg_list[0]

def calc_enemy(cl_pds,view, planars, elab_folder):
    pd_list = []
    for key in cl_pds:
        pd_list.append(cl_pds[key])
    cluster_pd_1D_match = pd.concat(pd_list)
    cluster_pd_1D_match = cluster_pd_1D_match[cluster_pd_1D_match[f"cl_pos_{view}"].notna()]
    enemy_res_list = []
    chi_list = []
    residual_fit_list = []
    pos_res_list = []
    error_list = []
    count_list = []
    couples = [(0, 1), (1, 2), (2, 3), (0,2), (1,3), (0,3)]
    couples = [couple for couple in couples if np.all(np.isin(couple, planars))]

    for pls in tqdm(couples, desc="Couples", leave=False):
        complete_evt = cluster_pd_1D_match.groupby("count").filter(lambda x: all([i in set(x.planar.values) for i in set(pls)]))
        residual_list = complete_evt.groupby("count", axis=0).apply(lambda x: x[x.planar == pls[0]][f"cl_pos_{view}_cm"].values[0] - x[x.planar == pls[1]][f"cl_pos_{view}_cm"].values[0])
        cluster_pd_1D_match[f"ene_{pls}"] = cluster_pd_1D_match["count"].apply(lambda x: assign_residual(x, residual_list))

        pos_list = complete_evt.groupby("count", axis=0).apply(lambda x: x[x.planar == pls[0]][f"cl_pos_{view}_cm"].values[0])
        count_list.append(complete_evt.groupby("count", axis=0).apply(lambda x: x[x.planar == pls[0]]["count"].values[0]))


        residual_list=residual_list[residual_list<20]
        sigma_def = r_fit.estimate_sigma_def(residual_list)
        popt_list, pcov_list, res_list, R_list, chi, deg_list, error = r_fit.single_gaus_fit_root(residual_list, sigma_def=sigma_def)
        popt_list.extend([0,1,1])
        residual_list_plot=pd.DataFrame(residual_list.rename(f"res_{view}"))
        plot = plot_residuals(residual_list_plot, view, popt_list, R_list, pls, chi, deg_list)
        plot[0].savefig(os.path.join(elab_folder, f"Enemy_gaus_fit_{pls}{view}.png"))
        plt.close(plot[0])
        enemy_res_list.append(popt_list[2])
        error_list.append(error[2])
        chi_list.append(chi/deg_list)
        residual_fit_list.append(residual_list)
        pos_res_list.append(pos_list)
    return couples,enemy_res_list, chi_list, residual_fit_list, pos_res_list, error_list, count_list, cluster_pd_1D_match

    def calc_enemy_align(self, view):
        # pd_list = []
        # for key in self.cl_pds:
        #     pd_list.append(self.cl_pds[key])
        # cluster_pd_1D_match = pd.concat(pd_list)
        # cluster_pd_1D_match = cluster_pd_1D_match[cluster_pd_1D_match[f"cl_pos_{view}"].notna()]
        # enemy_res_list = []
        # chi_list = []
        # enemey_res_list = []
        # pos_res_list = []
        # error_list = []
        # count_list = []
        # for pls in tqdm([(0, 1), (1, 2), (2, 3), (0,2), (1,3), (0,3)], desc="Couples", leave=False):
        #     complete_evt = cluster_pd_1D_match.groupby("count").filter(lambda x: all([i in set(x.planar.values) for i in set(pls)]))
        #     residual_list = complete_evt.groupby("count", axis=0).apply(lambda x: x[x.planar == pls[0]][f"cl_pos_{view}_cm"].values[0] - x[x.planar == pls[1]][f"cl_pos_{view}_cm"].values[0])
        #     pos_list = complete_evt.groupby("count", axis=0).apply(lambda x: x[x.planar == pls[0]][f"cl_pos_{view}_cm"].values[0])
        #     count_list.append(complete_evt.groupby("count", axis=0).apply(lambda x: x[x.planar == pls[0]]["count"].values[0]))
        #
        #     sigma_def = r_fit.estimate_sigma_def(residual_list)
        #     popt_list, pcov_list, res_list, R_list, chi, deg_list, error = r_fit.single_gaus_fit_root(residual_list, sigma_def=sigma_def)
        #     enemy_res_list.append(popt_list[2])
        #     error_list.append(error[2])
        #     chi_list.append(chi/deg_list)
        #     enemey_res_list.append(residual_list)
        #     pos_res_list.append(pos_list)
        #
        #
        # return enemy_res_list, chi_list, enemey_res_list, pos_res_list, error_list, count_list
        pass

def assign_residual(x, residual_list):
    """
    Function used to store the enemy residual data
    """
    if x in residual_list.index:
        return residual_list[x]
    else:
        return np.nan
def save_html_event(fig, save_path):
    """
    Save the html file from an event figure
    """
    fig.update_layout(
        width=750,
        height=480
    )
    fig.write_html(save_path, include_plotlyjs="directory")


def save_evt_display(run, data_folder, planar, nevents):
    print (f"Saving events, run {run}, planar{planar}\n")
    os.path.join(data_folder, "/perf_out/", f"{run}", )
    correction = perf.load_nearest_correction(os.path.join(data_folder, "alignment"), run)  # Load the alignment correction
    ## Loads the datafram for the selected planar
    cluster_pd_1D = pd.read_feather(os.path.join(data_folder, "raw_root", f"{run}", "cluster_pd_1D-zstd.feather"))
    hit_pd = pd.read_feather(os.path.join(data_folder, "raw_root", f"{run}", "hit_data-zstd.feather"))
    cluster_pd_1D_match = pd.read_feather(os.path.join(data_folder,"perf_out", f"{run}", f"match_cl_{planar}-zstd.feather" ))
    trk_pd = pd.read_pickle(os.path.join(data_folder,"perf_out", f"{run}", f"tracks_pd_{planar}.gzip"), compression = "gzip")
    eff_pd = pd.read_feather(os.path.join(data_folder,"perf_out", f"{run}", f"eff_pd_{planar}-zstd.feather" ))

    elab_folder= os.path.join(data_folder, "elaborated_output", f"{run}")
    ## Builds the folders
    if not os.path.isdir(elab_folder) :
        os.mkdir(elab_folder)

    evt_folder_eff = os.path.join(elab_folder, "events_eff")
    if not os.path.isdir(evt_folder_eff) :
        os.mkdir(evt_folder_eff)

    evt_folder_ineff = os.path.join(elab_folder, "events_ineff")
    if not os.path.isdir(evt_folder_ineff):
        os.mkdir(evt_folder_ineff)

    displ= event_visualizer(cluster_pd_1D, cluster_pd_1D_match, trk_pd,hit_pd, eff_pd, planar, correction)

    eff_evt=eff_pd[ (eff_pd.eff_x) & (eff_pd.eff_y)]["count"].unique()
    ineff_evt=eff_pd[ (~(eff_pd.eff_x) | ~(eff_pd.eff_y))]["count"].unique()

    for i in range (0, nevents):
        evt=np.random.choice(eff_evt)
        fig_x, fig_y = displ.plot_evt(evt)
        save_html_event(fig_x, os.path.join(evt_folder_eff, f"Evt_{evt}_planar_{planar}_x.html"))
        save_html_event(fig_y, os.path.join(evt_folder_eff, f"Evt_{evt}_planar_{planar}_y.html"))

        evt=np.random.choice(ineff_evt)
        fig_x, fig_y = displ.plot_evt(evt)
        save_html_event(fig_x, os.path.join(evt_folder_ineff, f"Evt_{evt}_planar_{planar}_x.html"))
        save_html_event(fig_y, os.path.join(evt_folder_ineff, f"Evt_{evt}_planar_{planar}_y.html"))

def extract_eff_and_res(run, data_folder, planar_list, tpc=False):
    """
    Calculate efficiency end resolution
    """
    if tpc:
        tpc_string = "_TPC"
    else:
        tpc_string = ""
    perf_path = os.path.join(data_folder, "perf_out", f"{run}")
    log_file = os.path.join(perf_path, "logfile")
    correction = perf.load_nearest_correction(os.path.join(data_folder, "alignment"), run)  # Load the alignment correction
    eff_pd_l = []
    for planar in planar_list:
        eff_pd = pd.read_feather(os.path.join(perf_path, f"eff_pd_{planar}"+tpc_string+"-zstd.feather" ))
        eff_pd = eff_pd[(eff_pd.pos_x > 4) & (eff_pd.pos_x < 7) & (eff_pd.pos_y > 4) & (eff_pd.pos_y < 7)]
        eff_pd_l.append(eff_pd)
    eff_pd = pd.concat(eff_pd_l)
    hit_pd = pd.read_feather(os.path.join(data_folder, "raw_root", f"{run}", "hit_data-zstd.feather"))

    elab_folder= os.path.join(data_folder, "elaborated_output", f"{run}")
    ## Builds the folders
    if not os.path.isdir(elab_folder):
        os.mkdir(elab_folder)
    ## Calc efficiency
    eff_calc = eff_calculation(eff_pd, hit_pd=hit_pd, log_path=log_file, correction=correction, outpath=elab_folder)
    eff_calc.calc_eff(planar_list=planar_list)

    ## Cal res
    trk_pd_l = {}
    cl_pd_l = {}
    for planar in planar_list:
        trk_pd_l[planar] = pd.read_pickle(os.path.join(data_folder,"perf_out", f"{run}", f"tracks_pd_{planar}"+tpc_string+".gzip" ), compression="gzip")
        cl_pd_l[planar] = pd.read_feather(os.path.join(data_folder,"perf_out", f"{run}", f"match_cl_{planar}"+tpc_string+"-zstd.feather" ))

    res_calc = res_measure(cl_pd=cl_pd_l, tracks_pd=trk_pd_l, eff_pd=eff_pd, planar_list=planar_list)
    logger = perf.log_writer(elab_folder, 0, "resolution.txt")

    for planar in planar_list:
        for view in ("x","y"):
            popt_list, pcov_list, res_list, R_list, chi_list, deg_list = res_calc.calc_res(planar, view)
            errror_tracking = (res_calc.cl_pds[f"{planar}{view}"].error_tracking**(1/2)).mean()
            plot = plot_residuals(res_calc.cl_pds[f"{planar}{view}"], view, popt_list, R_list, planar, chi_list, deg_list)
            plot[0].savefig(os.path.join(elab_folder, f"double_gaus_fit_{planar}{view}.png"))
            plt.close(plot[0])
            logger.write_log( f"Planar {planar} view {view}: Sigma_0={(popt_list[2]) * 10000:.2f} um, Sigma_1={popt_list[5] * 10000:.2f} um, "
                              f"error tracking: {errror_tracking * 10000:.2f} um" )
    ## Enemy
    logger.write_log(f"\n--Enemy residual--\n")
    s0 = sym.Symbol("s0", positive=True)
    s1 = sym.Symbol("s1", positive=True)
    s2 = sym.Symbol("s2", positive=True)
    s3 = sym.Symbol("s3", positive=True)
    sb = sym.Symbol("sb", positive=True)
    sc = sym.Symbol("sc", positive=True)


    for view in ("x","y"):
        logger.write_log(f"\n--Enemy residual {view}--\n")

        couples,enemy_res_list, chi_list, enemey_res_list, pos_res_list, error_list, count_list, enemy_pd = calc_enemy(res_calc.cl_pds,view, planar_list, elab_folder)
        enemy_pd.reset_index(drop=True, inplace=True)
        enemy_pd.drop(columns=["cov"], inplace=True)
        enemy_pd.to_feather(os.path.join(perf_path, f"match_cl_enemy_{view}"+tpc_string+"-zstd.feather"), compression="zstd")

        for couple, enemy_res in zip(couples, enemy_res_list):
            logger.write_log(f"Couple: {couple}: {enemy_res*10000:.2f} um")
        if len (couples)>4:
            sol=sym.solve([(s0**2+s1**2+sb**2)**(1/2)-enemy_res_list[0],
           (s1**2+s2**2+sb**2)**(1/2)-enemy_res_list[1],
           (s2**2+s3**2+sb**2)**(1/2)-enemy_res_list[2],
           (s0**2+s2**2+(2*sb)**2)**(1/2)-enemy_res_list[3],
           (s1**2+s3**2+(2*sb)**2)**(1/2)-enemy_res_list[4],
           (s0**2+s3**2+(sc)**2)**(1/2)-enemy_res_list[5]
              ],
             (s0, s1, s2,s3,sb, sc))
            if len(sol)>0:
                logger.write_log("System solution:\n"
                                 f"{sol[-1]}")
            else:
                sol = sym.solve([(s0 ** 2 + s1 ** 2 + sb ** 2) ** (1 / 2) - enemy_res_list[0],
                                 (s1 ** 2 + s2 ** 2 + sb ** 2) ** (1 / 2) - enemy_res_list[1],
                                 (s2 ** 2 + s3 ** 2 + sb ** 2) ** (1 / 2) - enemy_res_list[2],
                                 (s0 ** 2 + s2 ** 2 + (2 * sb) ** 2) ** (1 / 2) - enemy_res_list[3],
                                 (s0 ** 2 + s3 ** 2 + (3 * sb) ** 2) ** (1 / 2) - enemy_res_list[5]
                                 ],
                                (s0, s1, s2, s3, sb))
                if len(sol) > 0:
                    logger.write_log("System solution:\n"
                                     f"{sol[-1]}")
                else:
                    logger.write_log("Can't find solution:\n")


def plot_residuals(cl_pd_res, view, popt_list, R_list, pl, chi_list, deg_list):
    data = cl_pd_res[f"res_{view}"]
    data = data[abs(data) < 12]
    sigma_0 = r_fit.estimate_sigma_def(data)
    # data = data[abs(data) < sigma_def]
    # if data.shape[0] > 20000:
    #     nbins = 1000
    # else:
    #     nbins = 200
    nbins=200
    y, x = np.histogram(data, bins=nbins, range=[np.mean(data)-sigma_0, np.mean(data)+sigma_0])
    x = (x[1:] + x[:-1]) / 2
    # x = np.insert(x, 0, -0.2)
    # y = np.insert(y, 0, 0)
    popt = popt_list
    f, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b*', label='data')
    x = np.arange(np.min(x), np.max(x), 0.0002)
    ax.plot(x, r_fit.gaus(x, *popt[0:3]), 'c-', label='fit 0')
    ax.plot(x, r_fit.gaus(x, *popt[3:6]), 'g-', label='fit 1')
    ax.plot(x, r_fit.doublegaus(x, *popt), 'r-', label='fit cumulative')
    ax.grid()
    # plt.legend()
    # plt.title('Fig. 3 - Fit for Time Cons§tant')
    ax.set_ylabel('#')
    ax.set_xlabel('Residual [cm]')
    # plt.ion()
    # plt.show()
    ax.set_title(f"Fit view {view}, planar{pl}")
    ax.text(y=np.max(y) * 0.7, x=np.min(x)+0.001,
            s=f"R^2={R_list:.4f}\nNorm_0={popt[0]:.2f}, Mean_0={popt[1] * 10000:.2f}um, Sigma_0={(popt[2]) * 10000:.2f}um"
              f"\nNorm_1={popt[3]:.2f}, Mean_1={popt[4] * 10000:.2f}um, Sigma_1={abs(popt[5]) * 10000:.2f}um"
              f"\nChi_sqrt={chi_list:.3e}, Chi_sqrt/NDoF = {chi_list / deg_list:.3e}",
            fontsize="small")
    ax.set_xlim([np.min(x), np.max(x)])
    #     if put==pl:
    #         plt.savefig(os.path.join(os.path.join(path_out_eff, "res_fit"), f"fit_res_DUT_pl{pl}_DUT_{put}{view}.png"))
    #     else:
    #         plt.savefig(os.path.join(os.path.join(path_out_eff, "res_fit"), f"fit_res_TRK_pl{pl}_DUT_{put}{view}.png"))

    return f, ax
