import numpy as np
import pandas as pd
import os
import requests
from scipy.optimize import curve_fit
from multiprocessing import Pool
from tqdm import tqdm
import time
from scipy import special
import warnings
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import math
from scipy.odr import ODR, Model, Data, RealData
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import root_fit_lib
import plotly.io as pio
from planar_analysis_lib import charge_centroid
pio.templates.default = "plotly_white"

warnings.filterwarnings('ignore')


def time_walk_rough_corr(charge, signal_width, a, b, c):
    charge = np.array(charge)
    if charge <= 0:
        charge = 0.01
    time = a / charge ** b + c
    if time > signal_width / 4 + 60:
        time = signal_width / 4 + 60
    return time


# def calc_tpc_pos(cluster, hits, vel_l, ref_l):
#     """
#     Generic function to calc the tpc position on a cluster
#     :param cluster:
#     :param hits:
#     :param vel_l:
#     :param ref_l:
#     :return:
#     """
#     vel = vel_l[cluster.planar]
#     ref_time = ref_l[cluster.planar]
#
#     hits = hits[hits.hit_id.isin(cluster.hit_ids)]
#     hits["pos_g"] = (hits.hit_time - ref_time) * vel
#     if hits.strip_x.nunique() > 1:
#         try:
#             fit = np.polyfit(
#                 x=np.float64(hits.strip_x.values),
#                 y=np.float64(hits.pos_g.values),
#                 w=np.float64(1 / (hits.hit_time_error.values * vel)),
#                 deg=1
#             )
#             pos_utpc = (2.5 - fit[1]) / fit[0]
#         except ValueError as E:
#             print (ValueError, "Hits" , hits)
#             print (f"Count: {hits['count'].mean()} ")
#             pos_utpc=cluster.cl_pos_x
#         return pos_utpc
#     else:
#         return cluster.cl_pos_x


def time_walk_rough_corr_calib(charge, a, b, c):
    charge = np.array(charge)
    time = a / charge ** b + c
    return time


def calc_corr(row, dict_calibrations, thr_tmw):
    g = row.gemroc
    t = row.tiger
    c = row.channel
    return (time_walk_rough_corr(row.charge_SH, *dict_calibrations[thr_tmw[g, t, c]]))


def minus_errorfunc(x, x0, sig, c):
    y = (special.erf((x - x0) / (1.4142 * sig))) * c / 2 + 0.5 * c
    return -y


def errorfunc(x, x0, sig, c):
    y = (special.erf((x - x0) / (1.4142 * sig))) * c / 2 + 0.5 * c
    return y


def linear_func(p, x):
    """ Linear function to use like model for ODR fitting"""
    m, c = p
    return m * x + c


fit_model = Model(linear_func)


class tpc_prep:
    """
    Class to run TPC
    """

    def __init__(self, data_folder, cpu_to_use, run, cylinder, signal_width=80, silent=False,
                 ):
        self.data_folder = data_folder
        self.cpu_to_use = cpu_to_use
        self.run_number = run

        self.cylinder = cylinder
        self.signal_width = signal_width
        self.tpc_dir = os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"tpc")
        if not os.path.isdir(self.tpc_dir):
            os.mkdir(self.tpc_dir)
        self.cut = 0.15
        self.silent = silent
        self.no_errors = False
        self.no_first_last_shift = False
        self.no_capacitive = False
        self.drift_velocity = False
        self.no_time_walk_corr = False
        self.no_border_correction = False
        self.no_prev_strip_charge_correction = False
        self.fixed_angle = -1

        self.beta0 = [0.5, -20]  # initial guess
        self.ifixb = [1, 1]  # free parameter
        self.no_pos_g_cut = False
        self.no_big_clusters_splitting = False
        self.no_diffusion_error = False
        self.capacitive_cut_value = 0.2
        self.tpc_angle = 0


    def thr_tmw(self, row):
        """
        Extract the nearest calibration value to the thr_eff
        """
        thr_poss = np.array([0.5, 1, 2, 3])
        return thr_poss[np.argmin(abs(row - thr_poss))]

    def thr_eff(self, row):
        """
        Extract the thr_eff
        """
        if len(row) > 50:
            y, x = np.histogram(row.values, bins=500)
            return x[np.argmin(abs(y - np.max(y) / 2)[:np.argmax(y)])]
        else:
            return np.nan

    def get_calibration_time_walk_courve(self, time, thr):
        """
        Get the calibration curve from Fabio's repo
        """
        if thr > 0.5:
            url = f"https://raw.githubusercontent.com/fabio-cossio/TIGER/master/TimeWalk/{time}/timeWalk_{thr}f_{time}ns.txt"
        else:
            url = f"https://raw.githubusercontent.com/fabio-cossio/TIGER/master/TimeWalk/{time}/timeWalk_0f5_{time}ns.txt"
        page = requests.get(url)
        if "404: Not Found" in page.text:
            print(url)
            print("\nCalibration not found")
            return 0
        two_col = [row.split("\t") for row in page.text.split("\n")]
        x = [float(row[0]) for row in two_col[:-1]]
        y = [float(row[1]) for row in two_col[:-1]]
        popt, pcov = curve_fit(time_walk_rough_corr_calib, x, y)
        return popt

    def exctract_thr_eff(self):
        """
        Calculate and saves the effective thr, saves the nearest thr for the tmw calibration.
        """
        # hit_pd = pd.read_pickle(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"hit_data.pickle.gzip"), compression="gzip")
        hit_pd = pd.read_feather(
            os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"hit_data-zstd.feather"))
        hit_pd = hit_pd.query("charge_SH<10")
        thr_eff = hit_pd.groupby(["gemroc", "tiger", "channel"]).charge_SH.agg(self.thr_eff)
        thr_tmw = thr_eff.apply(self.thr_tmw)
        thr_eff.to_pickle(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"thr_eff.pickle"))
        thr_tmw.to_pickle(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"thr_tmw.pickle"))

    def apply_time_walk_corr_subrun(self, hit_pd):
        """
        Calculate and apply the time walk correction on a single subrun
        """
        thr_tmw = pd.read_pickle("/media/disk2T/VM_work_zone/data/raw_root/563/thr_tmw.pickle.gzip")
        dict_calibrations = {}
        for thr in (0.5, 1, 2, 3):
            dict_calibrations[thr] = np.append(np.array([self.signal_width]),
                                               self.get_calibration_time_walk_courve(self.signal_width, thr))
        hit_pd["hit_time_corr"] = hit_pd.apply(lambda x: calc_corr(x, dict_calibrations, thr_tmw), axis=1)
        # hit_pd["hit_time"] = -(hit_pd["l1ts_min_tcoarse"] - 1567) * 6.25 - 800
        hit_pd["hit_time"] = -(hit_pd["l1ts_min_tcoarse"] - 1567) * 6.25 - hit_pd["hit_time_corr"] - 800
        hit_pd["hit_time_error"] = ((hit_pd["hit_time_corr"] / 2) ** 2 + 6.25 / (12 ** 1 / 2) ** 2) ** (1 / 2)
        return hit_pd

    def apply_time_walk_corr_run(self):
        """

        """

        hit_pd = pd.read_feather(
            os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"hit_data-zstd.feather"))
        sub_list = []
        return_list = []
        hit_pd_sub = hit_pd.groupby(["subRunNo"])
        for key in hit_pd_sub.groups:
            sub_list.append(hit_pd_sub.get_group(key))
        if len(sub_list) > 0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(sub_list), desc="Calculating time and time walk", leave=False) as pbar:
                    for i, x in enumerate(pool.imap(self.apply_time_walk_corr_subrun, sub_list)):
                        return_list.append(x)
                        pbar.update()
        start = time.time()
        print("Concat")
        hit_pd = pd.concat(return_list, ignore_index=True)
        print(f"Time: {time.time() - start}")
        print("Sorting")

        hit_pd.sort_values(["subRunNo", "count", "hit_id"], inplace=True)
        hit_pd.reset_index(drop=True, inplace=True)
        hit_pd["hit_time_corr"] = hit_pd["hit_time_corr"].astype(np.float16)
        hit_pd["hit_time"] = hit_pd["hit_time"].astype(np.float16)
        print("Save")
        start = time.time()

        hit_pd.to_feather(os.path.join(self.tpc_dir, f"hit_data_wt-zstd.feather"), compression='zstd')
        # hit_pd.to_pickle(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"hit_data_wt.pickle.gzip"), compression="gzip")
        print(f"Time: {time.time() - start}")

    def calc_ref_time(self, hit_pd_c):
        """
        Calculate the reference time for the TPC fit
        :param hit_pd_c:
        :return:
        """
        y, x = np.histogram(hit_pd_c.hit_time, bins=100, range=[0, 625])
        x = x + 0.625 / 2
        x = x[:-1]
        # x=np.arange(1,625,6.25)
        # fig = px.scatter(y=y,x=x)
        y[y.argmax():] = y.max()
        fit, cov = curve_fit(errorfunc, x, y, p0=[150, 3, y.max()])
        ref_time = fit[0]
        return ref_time, fit

    def calc_drift_vel(self, hit_pd_c, ref_time):
        """
        Calculate the dirft velocity (! not precise for angles <30°)
        :param hit_pd_c:
        :param ref_time:
        :return:
        """
        y, x = np.histogram(hit_pd_c.hit_time, bins=100, range=[0, 625])
        x = x + 0.625 / 2
        x = x[:-1]
        y_max = np.argmax(y)
        y_der = np.gradient(y)
        y_df = savgol_filter(y_der, 3, 1)
        max_ind = argrelextrema(y_df, np.greater)
        cut_index = max_ind[0][max_ind[0] > y_max][0]
        y[:cut_index] = y[cut_index]
        # y[:min(range(len(x)), key=lambda i: abs(x[i]-200))] = y[min(range(len(x)), key=lambda i: abs(x[i]-206))]
        fit2, cov = curve_fit(minus_errorfunc, x, y, p0=[200, 3, y.max() / 2])
        vel = 5 / (fit2[0] - ref_time)
        return vel, fit2

    def plot_extraction(self, hit_pd_c, fit, fit2, ref_time, vel, pl):
        """
        Dave the plots of the drift velocity and time reference extraction
        :param hit_pd_c:
        :param fit:
        :param fit2:
        :param ref_time:
        :param vel:
        :param pl:
        :return:
        """
        y, x = np.histogram(hit_pd_c.hit_time, bins=100, range=[0, 625])
        x = x + 0.625 / 2
        x = x[:-1]
        figplot, ax = plt.subplots(figsize=(10, 8))
        ax.plot(x, y, "+", label="data")
        ax.plot(x, errorfunc(x, *fit), label="fit1")
        ax.plot(x, minus_errorfunc(x, *fit2), label="fit2")

        # plt.plot(y_der2 ,label= "ddy")
        ax.text(np.mean(x), np.mean(y), f"T_0 = {ref_time:.2f}, V_drift = {vel}")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("# hits")
        ax.legend()
        figplot.savefig(os.path.join(self.tpc_dir, f"fit_time_pl_{pl}.png"))

    def check_capacitive_border(self, event_hits, hit_pd_x):
        """
        Check if there is a capacitivly induced hit on the borders of the clusters
        """
        if event_hits.charge_SH.values[0] / event_hits.charge_SH.values[1] < self.cut:
            #         print ("capacitive coupling init")
            event_hits = event_hits[1:]
            hit_pd_x.loc[event_hits.index[0], "dropped"] = "Init"

        if event_hits.charge_SH.values[-1] / event_hits.charge_SH.values[-2] < self.cut:
            #         print ("capacitive coupling final")
            event_hits = event_hits[:-1]
            hit_pd_x.loc[event_hits.index[-1], "dropped"] = "Final"

        return (event_hits)

    def check_capacitive_border_two_strips(self, event_hits, hit_pd_x):
        """
        Checks for capacitive effects on the last and first 2 strips.
        """
        if event_hits.charge_SH[0:2].sum() / event_hits.charge_SH[3:4].sum() < self.cut:
            hit_pd_x.loc[event_hits.index[0:2], "dropped"] = "Init"
            event_hits = event_hits[2:]


        else:
            if event_hits.charge_SH.values[0] / event_hits.charge_SH.values[1] < self.cut:
                hit_pd_x.loc[event_hits.index[0], "dropped"] = "Init"
                event_hits = event_hits[1:]

        if event_hits.charge_SH[-2:].sum() / event_hits.charge_SH[-4:-2].sum() < self.cut:
            hit_pd_x.loc[event_hits.index[-2:], "dropped"] = "Final"
            event_hits = event_hits[:-2]



        else:
            if event_hits.charge_SH.values[-1] / event_hits.charge_SH.values[-2] < self.cut:
                hit_pd_x.loc[event_hits.index[-1], "dropped"] = "Final"
                event_hits = event_hits[:-1]

        return (event_hits)

    def split_big_clusters_in_x(self, input_data):
        """
        Split biggest cluster into smaller ones.
        :return:
        """
        angle = self.tpc_angle
        mean_size = ((np.tan(angle * np.pi / 180) * 5 + 0.35) // 0.65) + 1
        cluster_pd = input_data[0]
        hit_pd = input_data[1]
        big_clusters = cluster_pd[cluster_pd.cl_size > mean_size]
        hit_pd_x_big = hit_pd[hit_pd["count"].isin(big_clusters["count"])]
        for cl_index, cluster in big_clusters.iterrows():
            cluster_hits = hit_pd_x_big[hit_pd_x_big.hit_id.isin(cluster.hit_ids)]
            skipped_values = list(
                set(range(cluster_hits.strip_x.min(), cluster_hits.strip_x.max())) - set(cluster_hits.strip_x))
            skipped_values.sort()
            hits_list = []
            cluster_hits_remaining = cluster_hits
            for missing in skipped_values:
                hits_list.append(cluster_hits_remaining[cluster_hits_remaining.strip_x < missing])
                cluster_hits_remaining = cluster_hits_remaining[cluster_hits_remaining.strip_x > missing]
            hits_list.append(cluster_hits_remaining)
            model = cluster_pd[cluster_pd.index == cl_index].iloc[0]
            max_cl_id = cluster_pd[cluster_pd["count"] == model["count"]].cl_id.max()
            cluster_pd.drop(index=[cl_index], inplace=True)
            row = model
            for n, cl_hits in enumerate(hits_list):
                if cl_hits.shape[0] > 0:
                    row.loc["cl_pos_x"] = charge_centroid(cl_hits.strip_x, cl_hits.charge_SH)
                    row.loc["cl_charge"] = cl_hits.charge_SH.sum()
                    row.loc["cl_size"] = cl_hits.charge_SH.count()
                    row.loc["cl_id"] = max_cl_id + 1 + n
                    row.loc["hit_ids"] = cl_hits.hit_id.values
                    row.loc["planar"] = cl_hits.planar.values[0]
                    cluster_pd = cluster_pd.append(row)
        return cluster_pd

    def calc_tpc_pos_subrun(self, input_data):
        """
        Perform the tpc calculation on 1 subrun
        :return:
        """
        cluster_pd = input_data[0]
        hit_pd = input_data[1]
        cluster_pd_evts = cluster_pd.groupby("count")
        # hit_pd = self.hit_pd.query(f"subRunNo == {cluster_pd.subrun.mode().values[0]} and strip_x>-1")
        hit_pd["pos_g"] = np.nan
        hit_pd["dropped"] = "Not"
        hit_pd["strip_x_c"] = np.nan

        pitch = 0.650
        sx_coeff = (pitch / math.sqrt(12)) ** 2
        subrun = hit_pd.subRunNo.values[0]
        hit_pd_evts = hit_pd.groupby("count")
        # print (self.no_capacitive)
        for count in tqdm(cluster_pd_evts.groups, desc=f"Events subrun = {subrun}", leave=False,
                          position=(subrun % self.cpu_to_use + 1)):
            # for count in cluster_pd_evts.groups:
            clusters = cluster_pd_evts.get_group(count)
            events_hits = hit_pd_evts.get_group(count)
            for cl_index, cluster in clusters.iterrows():

                ref_time = self.ref_time_list[cluster.planar]
                if self.drift_velocity == 0:  ## Option to select the speed origin
                    vel = self.vel_list[cluster.planar]
                else:
                    vel = self.drift_velocity

                cluster_hits = events_hits[events_hits.hit_id.isin(cluster.hit_ids)]
                if self.no_time_walk_corr:  ## time walk correction option
                    cluster_hits["pos_g"] = (cluster_hits.hit_time + cluster_hits.hit_time_corr - ref_time) * vel
                else:
                    cluster_hits["pos_g"] = (cluster_hits.hit_time - ref_time) * vel
                if not self.no_pos_g_cut:
                    upper_cut=6
                    lower_cut=-1
                    hit_pd.loc[cluster_hits[(cluster_hits.pos_g > upper_cut) | (cluster_hits.pos_g < lower_cut) ].index, "pos_g_pre_cor"] = cluster_hits[(cluster_hits.pos_g > upper_cut) | (cluster_hits.pos_g < lower_cut) ].pos_g
                    hit_pd.loc[cluster_hits[(cluster_hits.pos_g > upper_cut) | (cluster_hits.pos_g < lower_cut) ].index, "pos_g"] = np.nan

                    cluster_hits = cluster_hits[(cluster_hits.pos_g < upper_cut) & (cluster_hits.pos_g > lower_cut) ]

                hit_pd.loc[cluster_hits.index, "pos_g_pre_cor"] = cluster_hits.pos_g
                hit_pd.loc[cluster_hits.index, "pos_g"] = cluster_hits.pos_g

                # cluster_hits = cluster_hits.query("charge_SH>0")  ## Taglia a carica >0
                cluster_hits["error_from_t"] = vel * 15 / (abs(cluster_hits.charge_SH) + 0.5)
                cluster_hits["error_from_diff"] = 0
                ## Capacitive corrections
                if not self.no_capacitive:  ## Capacitive correction option
                    if cluster_hits.shape[0] > 3:
                        cluster_hits = self.check_capacitive_border_two_strips(cluster_hits, hit_pd)
                avg_charge = cluster_hits.charge_SH.sum() / cluster_hits.charge_SH.shape[0]
                if cluster_hits.shape[0] > 1:
                    cluster_hits.loc[cluster_hits.index, "previous_strip_charge"] = cluster_hits.charge_SH.shift()
                    cluster_hits["charge_ratio_p"] = cluster_hits["charge_SH"] / cluster_hits["previous_strip_charge"]
                    if not self.no_prev_strip_charge_correction:
                        cluster_hits.loc[cluster_hits["charge_ratio_p"] < 1, "pos_g"] = cluster_hits[
                                                                                            "pos_g"] + 1.3 - 1.3 * \
                                                                                        cluster_hits["charge_ratio_p"]
                        hit_pd.loc[cluster_hits.index, "pos_g"] = cluster_hits.pos_g

                    if not self.no_errors:
                        error_x = np.sqrt(sx_coeff + sx_coeff * (avg_charge / cluster_hits.charge_SH))
                        error_y = cluster_hits.error_from_t.values
                    else:
                        error_x = cluster_hits.error_from_t.values * 0 + 1
                        error_y = cluster_hits.error_from_t.values * 0 + 1
                    pos_x_fit = np.float64(cluster_hits.strip_x.values) * pitch
                    pos_x_fit[0] = pos_x_fit[0] + self.first_last_shift
                    pos_x_fit[-1] = pos_x_fit[-1] - self.first_last_shift
                    hit_pd.loc[cluster_hits.index, "strip_x_c"] = pos_x_fit / pitch
                    # print (f"corr_pos{pos_x_fit / pitch}")
                    # print (f"pos_{cluster_hits.strip_x.values}")
                    data = RealData(pos_x_fit,
                                    np.float64(cluster_hits.pos_g.values),
                                    sx=error_x,
                                    sy=error_y)

                    # print (self.ifixb)
                    odr = ODR(data, fit_model, beta0=self.beta0, ifixb=self.ifixb)
                    out = odr.run()
                    fit = out.beta
                    # print (fit)

                    hit_pd.loc[cluster_hits.index, "error_x"] = error_x
                    hit_pd.loc[cluster_hits.index, "error_y"] = error_y
                    hit_pd.loc[cluster_hits.index, "residual_tpc"] = cluster_hits.pos_g - cluster_hits.strip_x * pitch * \
                                                                     fit[0] - fit[1]
                    hit_pd.loc[cluster_hits.index, "strip_min"] = cluster_hits.strip_x.values[0]
                    hit_pd.loc[cluster_hits.index, "strip_max"] = cluster_hits.strip_x.values[-1]
                    hit_pd.loc[cluster_hits.index, "next_strip_charge"] = cluster_hits.charge_SH.shift(-1)
                    hit_pd.loc[cluster_hits.index, "previous_strip_charge"] = cluster_hits.charge_SH.shift()
                    hit_pd.loc[cluster_hits.index, "total_charge"] = cluster_hits.charge_SH.sum()
                    hit_pd.loc[cluster_hits.index, "cl_size"] = cluster_hits.charge_SH.shape[0]
                    hit_pd.loc[cluster_hits.index, "residual_tpc_sum"] = np.sum(
                        cluster_hits.pos_g.values - cluster_hits.strip_x.values * pitch * fit[0] - fit[1])
                    cluster_pd.loc[cl_index, "pos_tpc"] = ((2.5 - fit[1]) / fit[0]) / 0.650
                    cluster_pd.loc[cl_index, "F0"] = fit[1]
                    cluster_pd.loc[cl_index, "F1"] = fit[0]

                    #         hit_pd.loc[]

                    ## Tagliando i bordi

        return hit_pd, cluster_pd

    def calc_tpc_pos(self, cpus=30):
        if self.fixed_angle>-1:
            print ("Using fixed angle")
            self.beta0 = [np.tan(self.fixed_angle*(np.pi/180)), -20]  # initial guess
            self.ifixb = [0, 1]  # free parameter

        hit_pd = pd.read_feather(os.path.join(self.tpc_dir, f"hit_data_wt-zstd.feather"))
        hit_pd = hit_pd.sort_values(["subRunNo", "count", "planar", "strip_x"]).reset_index(
            drop=True)  ## Sorting the values for later use
        if self.no_first_last_shift:
            self.first_last_shift = 0
        else:
            self.first_last_shift = 0.5
        self.vel_list = []
        self.ref_time_list = []
        if not self.silent:
            print("\n Calculation drift velocity and reference time")
        ### Calc ref time and vel
        for pl in tqdm(range(0, 4), desc="Planar", leave=False):
            cluster_pd_eff_cc = pd.read_feather(
                os.path.join(self.data_folder, "perf_out", f"{self.run_number}", f"match_cl_{pl}-zstd.feather"))
            ## Selecting only hit inside clusters (only in X)
            cluster_pd_eff_cc = cluster_pd_eff_cc.query(f"cl_pos_x>-2")
            hit_pd_c = hit_pd.query(f"planar=={pl}")
            hit_pd_c = hit_pd_c.query(f"strip_x>-1")
            total_mask = hit_pd_c.charge_SH > 1000
            for subrun in tqdm(cluster_pd_eff_cc.subrun.unique(), leave=False):
                ids = np.concatenate(cluster_pd_eff_cc.query(f"subrun=={subrun}").hit_ids.values)
                counts = cluster_pd_eff_cc.query(f"subrun=={subrun}")["count"].values
                total_mask = total_mask | (
                        (hit_pd_c.hit_id.isin(ids)) & (hit_pd_c.subRunNo == subrun) & (hit_pd_c["count"].isin(counts))
                )
            hit_pd_c["in_cl"] = total_mask
            hit_pd_c = hit_pd_c.query("in_cl")
            hit_pd_c = hit_pd_c.query(f"planar=={pl} and strip_x>-1 and charge_SH>10 and hit_time>0 and hit_time<500")
            ref_time, fit = self.calc_ref_time(hit_pd_c)
            vel, fit2 = self.calc_drift_vel(hit_pd_c, ref_time)
            self.plot_extraction(hit_pd_c, fit, fit2, ref_time, vel, pl)
            self.vel_list.append(vel)
            print (f"Calculate velocity pl {pl}: {vel}")
            self.ref_time_list.append(ref_time)
        cluster_pd = pd.read_feather(
            os.path.join(self.data_folder, "raw_root", f"{self.run_number}", "cluster_pd_1D-zstd.feather"))
        cluster_pd_y = cluster_pd.query("cl_pos_y>-1")
        cluster_pd = cluster_pd.query("cl_pos_x>-1")
        sub_data = cluster_pd.groupby(["subrun"])
        hit_pd_sub = hit_pd.groupby(["subRunNo"])
        sub_list = []
        return_list_cl = []
        return_list_hits = []

        for key in sub_data.groups:
            sub_list.append([sub_data.get_group(key), hit_pd_sub.get_group(key)])
        if len(sub_list) > 0:
            with Pool(processes=cpus) as pool:
                with tqdm(total=len(sub_list), desc="TPC pos calculation ", leave=False) as pbar:
                    for i, x in enumerate(pool.imap_unordered(self.calc_tpc_pos_subrun, sub_list)):
                        return_list_cl.append(x[1])
                        return_list_hits.append(x[0])
                        pbar.update()
        return_list_cl.append(cluster_pd_y)
        cluster_pd_micro = pd.concat(return_list_cl)
        hit_pd_micro = pd.concat(return_list_hits)

        cluster_pd_micro.reset_index(inplace=True, drop=True)
        hit_pd_micro.reset_index(inplace=True, drop=True)
        # print (cluster_pd_micro.dtypes)
        cluster_pd_micro.to_feather(os.path.join(self.tpc_dir, f"cluster_pd_1D_TPC-zstd.feather"), compression='zstd')
        hit_pd_micro.to_feather(os.path.join(self.tpc_dir, f"hit_pd_TPC-zstd.feather"), compression='zstd')
        pd_2d_return_list = []
        sub_list = []
        sub_data = cluster_pd_micro.groupby(["subrun"])
        for key in sub_data.groups:
            sub_list.append(sub_data.get_group(key))
        if not self.silent:
            print("Clusters 2-D")
        if len(sub_list) > 0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(sub_list), disable=self.silent) as pbar_2:
                    for i, x in enumerate(pool.imap_unordered(self.build_2D_clusters, sub_list)):
                        pd_2d_return_list.append(x)
                        pbar_2.update()
        print("Concatenating")
        cluster_pd_2D = pd.concat(pd_2d_return_list)
        print("Saving")
        cluster_pd_2D.reset_index(inplace=True, drop=True)
        cluster_pd_2D.to_feather(os.path.join(self.tpc_dir, f"cluster_pd_2D_TPC-zstd.feather"), compression='zstd')

    def build_2D_clusters(self, cluster_pd):
        dict_4_pd = {
            "run": [],
            "subrun": [],
            "count": [],
            "planar": [],
            "cl_pos_x": [],
            "cl_pos_x_tpc": [],
            "cl_pos_y": [],
            "cl_charge": [],
            "cl_charge_x": [],
            "cl_charge_y": [],
            "cl_size_x": [],
            "cl_size_y": [],
            "cl_size_tot": []
        }
        events_pd_clusters = cluster_pd.groupby(["count", "planar"])
        for key in events_pd_clusters.groups:
            event_pd_cl = events_pd_clusters.get_group(key)
            cls_x = event_pd_cl[event_pd_cl.cl_pos_x.notna()]
            cls_y = event_pd_cl[event_pd_cl.cl_pos_y.notna()]
            if (cls_x.shape[0] > 0) and (cls_y.shape[0] > 0):
                cl_x = event_pd_cl.loc[cls_x.cl_charge.idxmax(axis=0)]
                cl_y = event_pd_cl.loc[cls_y.cl_charge.idxmax(axis=0)]
                dict_4_pd["run"].append(self.run_number)
                dict_4_pd["subrun"].append(cl_x.subrun)
                dict_4_pd["count"].append(key[0])
                dict_4_pd["planar"].append(key[1])
                dict_4_pd["cl_pos_x"].append(cl_x.cl_pos_x)
                dict_4_pd["cl_pos_x_tpc"].append(cl_x.pos_tpc)
                dict_4_pd["cl_pos_y"].append(cl_y.cl_pos_y)
                dict_4_pd["cl_charge"].append(cl_x.cl_charge + cl_y.cl_charge)
                dict_4_pd["cl_charge_x"].append(cl_x.cl_charge)
                dict_4_pd["cl_charge_y"].append(cl_y.cl_charge)
                dict_4_pd["cl_size_x"].append(cl_x.cl_size)
                dict_4_pd["cl_size_y"].append(cl_y.cl_size)
                dict_4_pd["cl_size_tot"].append(cl_x.cl_size + cl_y.cl_size)
        return (pd.DataFrame(dict_4_pd))



from perf import load_nearest_correction, apply_correction
from perf_out_evt_display import res_measure


class plotter_after_tpc():
    def __init__(self, data_folder, run_number, angle):
        self.data_folder = data_folder
        self.run = run_number
        if not os.path.isdir(os.path.join(f"{self.data_folder}", "elaborated_output", f"{self.run}")):
            os.mkdir(os.path.join(f"{self.data_folder}", "elaborated_output", f"{self.run}"))
        self.out_path = os.path.join(f"{self.data_folder}", "elaborated_output", f"{self.run}", "output_plot_TPC")
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)
        # self.tracks_pd = pd.read_pickle(
        #     os.path.join(data_folder, "perf_out", f"{self.run}", f"tracks_pd_{2}.gzip"),
        #     compression="gzip")

        self.correction = load_nearest_correction(os.path.join(data_folder, "alignment"), self.run)
        self.hit_pd_x = pd.read_feather(
            os.path.join(data_folder, "raw_root", f"{self.run}", "tpc", "hit_pd_TPC-zstd.feather"))
        self.cl_pd_x = pd.read_feather(
            os.path.join(data_folder, "raw_root", f"{self.run}", "tpc", "cluster_pd_1D_TPC-zstd.feather"))
        self.hit_pd_x = self.hit_pd_x.query("strip_x>-1")
        self.hit_pd_x["strip_i"] = self.hit_pd_x["strip_x"]-self.hit_pd_x["strip_min"]
        self.cl_pd_x = self.cl_pd_x.query("cl_pos_x>-1")
        self.cl_pd_x_g = self.cl_pd_x.groupby(["count", "planar"])
        self.hit_pd_x_g = self.hit_pd_x.groupby(["count", "planar"])
        self.pitch = 0.650
        self.angle = angle
        cl_pd_list = []
        self.track_pd_list = []
        eff_pd_l = []
        for p in range(0, 4):
            cl_pd_list.append(
                pd.read_feather(f"/media/disk2T/VM_work_zone/data/perf_out/{self.run}/match_cl_{p}_TPC-zstd.feather"))
            self.track_pd_list.append(
                pd.read_pickle(f"/media/disk2T/VM_work_zone/data/perf_out/{self.run}/tracks_pd_{p}_TPC.gzip",
                               compression="gzip"))
            eff_pd_l.append(
                pd.read_feather(f"/media/disk2T/VM_work_zone/data/perf_out/{self.run}/eff_pd_{p}_TPC-zstd.feather"))
        self.res_measure = res_measure(cl_pd=cl_pd_list, eff_pd=pd.concat(eff_pd_l), planar_list=[0, 1, 2, 3],
                                       tracks_pd=self.track_pd_list)

        self.cl_pd_x_enemy = pd.read_feather(
            os.path.join(data_folder, "perf_out", f"{self.run}","match_cl_enemy_x_TPC-zstd.feather"))
        self.cl_pd_y_enemy = pd.read_feather(
            os.path.join(data_folder, "perf_out", f"{self.run}","match_cl_enemy_y_TPC-zstd.feather"))

    def apply_correction_x(self, cl_pos_x_cm, epos_y, planar):
        """
        Fucntion used to apply 2D correction to the dataset
        :param row:
        :param planar:
        :param correction:
        :return:
        """
        corrections = self.correction
        for correction in corrections:
            angle = (correction[f"{int(planar)}_x"][0] - correction[f"{int(planar)}_y"][0]) / 2
            cl_pos_x_cm = cl_pos_x_cm - angle * (epos_y) + correction[f"{int(planar)}_y"][1]
        return cl_pos_x_cm

    def plot_evt_tpc(self, count, dut, folder):
        event_hits = self.hit_pd_x_g.get_group((count, dut))
        event_cluster = self.cl_pd_x_g.get_group((count, dut))
        event_cluster = event_cluster.loc[event_cluster["cl_charge"].idxmax()]
        event_hits = event_hits[event_hits.hit_id.isin(event_cluster.hit_ids)]
        event_hits = event_hits[~event_hits.pos_g.isna()]
        x = event_hits.strip_x_c
        y = event_hits.pos_g
        y_no_tw = event_hits.pos_g + event_hits.hit_time_corr * 0.040
        sx = event_hits.error_x
        sy = event_hits.error_y
        fit = [event_cluster.F1, event_cluster.F0]
        fit_x = self.track_pd_list[dut].query(f"count == {count}").fit_x.values[0]
        fit_y = self.track_pd_list[dut].query(f"count == {count}").fit_y.values[0]
        prev_pos = fit_x[1] + fit_x[0] * dut * 10
        res = self.res_measure.cl_pds[f'{dut}x'].query(f"count == {count}").res_x.min()

        figplot, ax = plt.subplots(2, 2, figsize=(20, 20))
        figplot.suptitle(f"Event {int(count)}, planar {dut}, "
                         f"residual : {res*10:.2f} mm"
                         , fontsize=26)

        figplot.set_facecolor("white")
        pos_utpc = event_cluster.pos_tpc
        ax[0][0].errorbar(
            x=x, y=y, yerr=sy, xerr=sx, fmt='o', label="TW correction", markersize=0.1)
        ax[0][0].errorbar(
            x=x, y=y_no_tw, yerr=sy, xerr=sx, fmt='o', label="No TW correction", markersize=0.1)

        ax[0][0].plot(x, (fit[1] + x * 0.650 * fit[0]), "r--", label="Fit")
        ax[0][0].plot([x.min(), x.max()], [0, 0], "b--")
        ax[0][0].plot([x.min(), x.max()], [5, 5], "b--")

        # ax[0][0].plot([pos_utpc_corr/0.0650,pos_utpc_corr/0.0650], [0, 5], "y--", label="Pos_utcp corrected")
        ax[0][0].plot([pos_utpc, pos_utpc], [0, 5], "r--", label="Pos_utcp")

        # ax[0][0].plot([prev_pos/0.0650,prev_pos/0.0650], [0, 5], "y-", label="prev_pos corrected")
        pos_utpc_corr = self.apply_correction_x(pos_utpc * 0.0650, fit_y[1] + fit_y[0] * dut * 10, planar=dut)
        corr = (pos_utpc_corr / 0.0650) - pos_utpc

        ax[0][0].plot([prev_pos / 0.0650 - corr, prev_pos / 0.0650 - corr], [0, 5], "g-", label="prev_pos")
        ## Retta passante per y = 2.5, x= prev_pos/0.0650
        ax[0][0].plot([((prev_pos - corr * 0.0650) - np.tan(self.angle * np.pi / 180) * 0.25) / 0.0650,
                       ((prev_pos - corr * 0.0650) + np.tan(self.angle * np.pi / 180) * 0.25) / 0.0650],
                      [0, 5], "g-", label="Prev track")

        ax[0][0].plot([x.min(), x.max()],
                      [2.5, 2.5], "k--", label="Half gap")
        ## Adding correction notes~~~~~~~~~~~~:
        dropped = event_hits.query('dropped!="Not"')
        ax[0][0].scatter(x=dropped.strip_x, y=dropped.pos_g, label="Capacitive corr", marker='x', c="green", s=400)

        ## Adding correction notes~~~~~~~~~~~~:

        ax2 = ax[0][0].twinx()
        ax2.bar(event_hits.strip_x, event_hits.charge_SH.values, width=0.4, color="cyan", alpha=0.1)
        ax[0][0].set_zorder(1)
        ax[0][0].patch.set_visible(False)
        ax[0][0].set_xlabel("Strip")
        ax2.set_ylabel("Charge [fC]")
        ax[0][0].legend()

        # --------
        ax[0][1].scatter(
            x=event_hits.strip_x, y=event_hits.pos_g_pre_cor, marker='+', label="Pre-correction", s=100, alpha=0.7)
        ax[0][1].scatter(
            x=event_hits.strip_x_c, y=event_hits.pos_g, marker='x', label="Post-correction", s=100, alpha=0.7)

        ax3 = ax[0][1].twinx()
        ax3.bar(event_hits.strip_x, event_hits.charge_SH.values, width=0.4, color="cyan", alpha=0.1)
        ax[0][1].set_zorder(1)
        ax[0][1].patch.set_visible(False)
        ax[0][1].set_xlabel("Strip")
        ax3.set_ylabel("Charge [fC]")
        ax[0][1].legend()
        # -------
        res = y - ((fit[1] + x * 0.650 * fit[0]))
        ax[1][0].errorbar(
            x=x, y=res, fmt='o', label="Residual")

        ax[1][0].set_xlabel("Strip")
        ax[1][0].set_ylabel("Res")

        ax[1][1].errorbar(
            x=y / 0.42, y=res, fmt='o', label="Residual")
        # ax[0][0].set_yticks(np.arange(y.min()//1, y.max(), 0.5))
        ax[1][1].set_xlabel("Time [ns]")
        ax[1][1].set_ylabel("Res")

        figplot.savefig(os.path.join(folder, f"evt_{count}_detector_{dut}.png"))

    def save_evts_plots(self, n=5):
        for pl in range(0, 4):
            residuals = self.res_measure.cl_pds[f"{pl}x"].res_x  # Load residuals
            residuals = residuals[residuals < 5]  # cut for absurd residuals
            std = residuals.std()
            print(std)
            good_evts_res = self.res_measure.cl_pds[f"{pl}x"][abs(self.res_measure.cl_pds[f"{pl}x"].res_x) < std / 2]
            bad_evts_res = self.res_measure.cl_pds[f"{pl}x"][(abs(self.res_measure.cl_pds[f"{pl}x"].res_x) > std) & (
                    abs(self.res_measure.cl_pds[f"{pl}x"].res_x) < std * 3)]
            very_bad_evts_res = self.res_measure.cl_pds[f"{pl}x"][
                abs(self.res_measure.cl_pds[f"{pl}x"].res_x) > std * 4]

            good_evts = np.random.choice(good_evts_res["count"], n)
            if bad_evts_res.shape[0]>n:
                bad_evts = np.random.choice(bad_evts_res["count"], n)  # Select good and bad evts
            if very_bad_evts_res.shape[0]>n:
                very_bad_evts = np.random.choice(very_bad_evts_res["count"], n)  # Select good and bad evts

            ## Folders creation
            good_folder = os.path.join(self.out_path, "good_evt")
            bad_folder = os.path.join(self.out_path, "bad_evt")
            very_bad_folder = os.path.join(self.out_path, "very_bad_evts")
            if not os.path.exists(good_folder):
                os.makedirs(good_folder)
            if not os.path.exists(bad_folder):
                os.makedirs(bad_folder)
            if not os.path.exists(very_bad_folder):
                os.makedirs(very_bad_folder)
            ## Plotto eventi buoni e cattivi
            for evt in good_evts:
                self.plot_evt_tpc(evt, pl, good_folder)
            for evt in bad_evts:
                self.plot_evt_tpc(evt, pl, bad_folder)
            for evt in very_bad_evts:
                self.plot_evt_tpc(evt, pl, very_bad_folder)

    def produce_plots(self):
        self.plt_path = os.path.join(self.out_path, "plots")
        if not os.path.isdir(self.plt_path):
            os.mkdir(self.plt_path)

        self.plot_residuals()
        self.plot_residuals_w_fit()
        self.plot_residuals_vs_pos()
        self.plot_residual_vs_charge()
        self.plot_residual_vs_size()
        self.plot_residual_vs_diff_with_cc()
        self.plot_residual_vs_fit_angle()
        self.plot_residual_vs_angle_track()
        self.plot_angle_fit_vs_angle_track()

        self.plot_residual_TPC_vs_charge()
        self.plot_residual_TPC_vs_pos_g()
        self.plot_residual_TPC_vs_near_strip_charge("next_strip_charge")
        self.plot_residual_TPC_vs_near_strip_charge("previous_strip_charge")
        self.plot_residual_TPC_vs_strip_index()
        self.plot_residual_TPC_vs_near_strip_ratio("next_strip_charge")
        self.plot_residual_TPC_vs_near_strip_ratio("previous_strip_charge")

        self.plot_enemy_angle_track()
        self.plot_enemy_angle_fit()
        self.plot_enemy_charge()

    ## Plot about residuals

    def plot_residuals(self):
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Detector 0", "Detector 1", "Detector 2", "Detector 3"),
                            horizontal_spacing=0.10)

        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram(x=self.res_measure.cl_pds[f"{pl}x"].res_x,
                             xbins={"start": -0.8, "end": 0.8, "size": 1.6 / 200},
                             name=f"Planare {pl}"),
                col=pl // 2 + 1, row=pl % 2 + 1)
        fig.update_xaxes(range=[-0.5, 0.5], dtick=0.1)
        fig.write_html(os.path.join(self.plt_path, "residuals.html"), include_plotlyjs="directory")
        # fig.show("notebook")

    def plot_residuals_w_fit(self):
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Detector 0", "Detector 1", "Detector 2", "Detector 3"),
                            horizontal_spacing=0.10)
        for pl in range(0, 4):
            popt, chi2, error, ndof = root_fit_lib.single_root_fit(self.res_measure.cl_pds[f"{pl}x"].res_x,
                                                                   p0=[100, 0, 0.1, 1],
                                                                   lower_bounds=[0, -2, 0, 0],
                                                                   upper_bounds=[5000, 2, 1, 1000], sigma_def=1)
            y, x = np.histogram(self.res_measure.cl_pds[f"{pl}x"].res_x, bins=200, range=[-1, 1])
            x = (x[1:] + x[:-1]) / 2
            fig.add_trace(
                go.Scatter(x=x, y=y,
                           line=dict(width=2, shape='hvh'), name=f"Detector {pl}"),
                col=pl // 2 + 1, row=pl % 2 + 1)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(-1, 1, 0.01),
                    y=root_fit_lib.gaus(np.arange(-1, 1, 0.01), *popt[0:3]) + popt[3],
                    name=f"Fit {pl}, m={popt[1]:.2e}, s={popt[2]:.2e}, cr={chi2 / ndof:.1e}"
                ), col=pl // 2 + 1, row=pl % 2 + 1)
        fig.update_xaxes(title="Res x [cm]")
        fig.update_yaxes(title="#")

        fig.update_xaxes(range=[-0.5, 0.5], dtick=0.1)

        fig.update_layout(legend=dict(
            orientation="h"
        ))
        fig.write_html(os.path.join(self.plt_path, "residuals_w_fit.html"), include_plotlyjs="directory")

    def plot_residuals_vs_pos(self):
        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram2d(x=self.res_measure.cl_pds[f"{pl}x"].cl_pos_x * 0.0650,
                               y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                               ybins={"start": -0.8, "end": 0.8, "size": 1.6 / 200},
                               xbins={"start": -1, "end": 10, "size": 10 / 128},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            fig.add_trace(
                go.Box(x=self.res_measure.cl_pds[f"{pl}x"].cl_pos_x * 0.0650 // 0.5 * 0.5,
                       y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                       name=f"Box det {pl}"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram(x=self.res_measure.cl_pds[f"{pl}x"].cl_pos_x * 0.0650 // 0.5 * 0.5,
                             name=f"Hist det {pl}", opacity=0.2),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=[-1, 9], title="Pos x [cm]")
        fig.update_yaxes(range=[-0.5, 0.5], title="Res x [cm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(width=1400, height=2000)

        fig.write_html(os.path.join(self.plt_path, "residuals_vs_pos_x.html"), include_plotlyjs="directory")

    def plot_residual_vs_charge(self):
        x_range = [0, 200]
        y_range = [-0.5, 0.5]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram2d(x=self.res_measure.cl_pds[f"{pl}x"].cl_charge,
                               y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                               ybins={"start": y_range[0], "end": y_range[1], "size": (y_range[1] - y_range[0]) / 100},
                               xbins={"start": x_range[0], "end": x_range[1], "size": (x_range[1] - x_range[0]) / 100},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            fig.add_trace(
                go.Box(x=self.res_measure.cl_pds[f"{pl}x"].cl_charge // 10 * 10,
                       y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                       name=f"Box det {pl}"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram(x=self.res_measure.cl_pds[f"{pl}x"].cl_charge,
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title="Cl charge [fC]")
        fig.update_yaxes(range=y_range, title="Res x [cm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)

        fig.write_html(os.path.join(self.plt_path, "residuals_vs_charge.html"), include_plotlyjs="directory")

    def plot_residual_vs_size(self):
        x_range = [0, 13]
        y_range = [-0.5, 0.5]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram2d(x=self.res_measure.cl_pds[f"{pl}x"].cl_size,
                               y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                               ybins={"start": y_range[0], "end": y_range[1], "size": (y_range[1] - y_range[0]) / 100},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / x_range[1]},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            fig.add_trace(
                go.Box(x=self.res_measure.cl_pds[f"{pl}x"].cl_size,
                       y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                       name=f"Box det {pl}"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram(x=self.res_measure.cl_pds[f"{pl}x"].cl_size,
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, dtick=1, title="Cl size [strips]")
        fig.update_yaxes(range=y_range, title="Res x [cm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)

        fig.write_html(os.path.join(self.plt_path, "residuals_vs_size.html"), include_plotlyjs="directory")

    def plot_residual_vs_diff_with_cc(self):
        x_range = [-2, 2]
        y_range = [-0.5, 0.5]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram2d(x=(self.res_measure.cl_pds[f"{pl}x"].cl_pos_x - self.res_measure.cl_pds[
                    f"{pl}x"].pos_tpc) * self.pitch,
                               y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                               ybins={"start": y_range[0], "end": y_range[1], "size": (y_range[1] - y_range[0]) / 100},
                               xbins={"start": x_range[0], "end": x_range[1], "size": (x_range[1] - x_range[0]) / 100},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            fig.add_trace(
                go.Box(x=((self.res_measure.cl_pds[f"{pl}x"].cl_pos_x - self.res_measure.cl_pds[
                    f"{pl}x"].pos_tpc) * self.pitch // 0.10) * 0.10,
                       y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                       name=f"Box det {pl}"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram(x=(self.res_measure.cl_pds[f"{pl}x"].cl_pos_x - self.res_measure.cl_pds[
                    f"{pl}x"].pos_tpc) * self.pitch,
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title="Diff btw CC and TPC [mm]")
        fig.update_yaxes(range=y_range, title="Res x [cm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)

        fig.write_html(os.path.join(self.plt_path, "residuals_vs_diff.html"), include_plotlyjs="directory")

    def plot_residual_vs_fit_angle(self):
        x_range = [-10, 90]
        y_range = [-0.5, 0.5]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram2d(x=(180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].F1),
                               y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 100},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 150},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            fig.add_trace(
                go.Box(x=((180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].F1) // 5) * 5,
                       y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                       name=f"Box det {pl}"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram(x=((180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].F1) // 5) * 5,
                             y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title="Fit angle [°]")
        fig.update_yaxes(range=y_range, title="Res x [cm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, "residuals_vs_fit_angle.html"), include_plotlyjs="directory")
    def plot_residual_vs_angle_track(self):
        x_range = [0-2, 0+2]
        y_range = [-0.5, 0.5]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram2d(x=(180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].angle_trk_x),
                               y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 100},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 150},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            fig.add_trace(
                go.Box(x=((180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].angle_trk_x) // 0.2) * 0.2,
                       y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                       name=f"Box det {pl}"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram(x=((180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].angle_trk_x) ) ,
                             y=self.res_measure.cl_pds[f"{pl}x"].res_x,
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title="Track_angle [°]")
        fig.update_yaxes(range=y_range, title="Res x [cm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, "residuals_vs_track_angle.html"), include_plotlyjs="directory")

    def plot_angle_fit_vs_angle_track(self):
        x_range = [0-2, 0+2]
        y_range = [-10, 90]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram2d(x=(180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].angle_trk_x),
                               y=(180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].F1),
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 100},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 150},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            fig.add_trace(
                go.Box(x=((180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].angle_trk_x) // 0.2) * 0.2,
                       y=(180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].F1),
                       name=f"Box det {pl}"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            fig.add_trace(
                go.Histogram(x=((180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].angle_trk_x) ) ,
                             y=(180 / np.pi) * np.arctan(self.res_measure.cl_pds[f"{pl}x"].F1),
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title="Track angle [°]")
        fig.update_yaxes(range=y_range, title="Fit angle [°]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, "fit_angle_vs_track_angle.html"), include_plotlyjs="directory")

    ## Plots about TPC fit residuals ##################################################################################

    def plot_residual_TPC_vs_charge(self):
        x_range = [-1, 50]
        y_range = [-2, 2]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]
            fig.add_trace(
                go.Histogram2d(x=hit_pd_c.charge_SH,
                               y=hit_pd_c.residual_tpc,
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 80},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 80},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]

            fig.add_trace(
                go.Box(x=(hit_pd_c.charge_SH // 5) * 5,
                       y=hit_pd_c.residual_tpc,
                       name=f"Box det {pl}", boxpoints=False),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]

            fig.add_trace(
                go.Histogram(x=hit_pd_c.charge_SH,
                             y=hit_pd_c.residual_tpc,
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title="Hit charge [fC]")
        fig.update_yaxes(range=y_range, title="Res fit TPC [mm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, "residuals_tpc_vs_hit_charge.html"), include_plotlyjs="directory")

    def plot_residual_TPC_vs_pos_g(self):
        x_range = [-5, 15]
        y_range = [-2, 2]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]
            fig.add_trace(
                go.Histogram2d(x=hit_pd_c.pos_g,
                               y=hit_pd_c.residual_tpc,
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 80},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 80},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]

            fig.add_trace(
                go.Box(x=hit_pd_c.pos_g // 1,
                       y=hit_pd_c.residual_tpc,
                       name=f"Box det {pl}", boxpoints=False),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]
            fig.add_trace(
                go.Histogram(x=hit_pd_c.pos_g,
                             y=hit_pd_c.residual_tpc,
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title="Pos in gap [mm]")
        fig.update_yaxes(range=y_range, title="Res fit TPC [mm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, "residuals_tpc_vs_pos_g.html"), include_plotlyjs="directory")

    def plot_residual_TPC_vs_near_strip_charge(self, field=""):
        x_range = [-1, 60]
        y_range = [-2, 2]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]
            fig.add_trace(
                go.Histogram2d(x=hit_pd_c[f"{field}"],
                               y=hit_pd_c.residual_tpc,
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 80},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 80},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]

            fig.add_trace(
                go.Box(x=(hit_pd_c[f"{field}"] // 5) *5,
                       y=hit_pd_c.residual_tpc,
                       name=f"Box det {pl}", boxpoints=False),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]
            fig.add_trace(
                go.Histogram(x=hit_pd_c[f"{field}"],
                             y=hit_pd_c.residual_tpc,
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title=f"{field} [fC]")
        fig.update_yaxes(range=y_range, title="Res fit TPC [mm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, f"residuals_tpc_vs_{field}.html"), include_plotlyjs="directory")
    def plot_residual_TPC_vs_strip_index(self):
        x_range = [-1, 15]
        y_range = [-2, 2]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]
            fig.add_trace(
                go.Histogram2d(x=hit_pd_c["strip_i"],
                               y=hit_pd_c.residual_tpc,
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 80},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 15},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]

            fig.add_trace(
                go.Box(x=hit_pd_c["strip_i"],
                       y=hit_pd_c.residual_tpc,
                       name=f"Box det {pl}", boxpoints=False),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]
            fig.add_trace(
                go.Histogram(x=hit_pd_c["strip_i"],
                             y=hit_pd_c.residual_tpc,
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range,  title=f"Strip i ")
        fig.update_yaxes(range=y_range, title="Res fit TPC [mm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, f"residuals_tpc_vs_strip_i.html"), include_plotlyjs="directory")

    def plot_residual_TPC_vs_near_strip_ratio(self, field=""):

        x_range = [0, 5]
        y_range = [-2, 2]
        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3",
                                "Detector 0", "Detector 1", "Detector 2", "Detector 3"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]
            fig.add_trace(
                go.Histogram2d(x=hit_pd_c[f"{field}"]/hit_pd_c.charge_SH,
                               y=hit_pd_c.residual_tpc,
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 80},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 80},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]

            fig.add_trace(
                go.Box(x=((hit_pd_c[f"{field}"]/hit_pd_c.charge_SH )// 0.5) *0.5,
                       y=hit_pd_c.residual_tpc,
                       name=f"Box det {pl}", boxpoints=False),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 4):
            hit_pd_c = self.hit_pd_x.query(f"planar == {pl}")
            hit_pd_c = hit_pd_c[hit_pd_c["count"].isin(np.random.choice(hit_pd_c["count"].unique(), 10000))]
            fig.add_trace(
                go.Histogram(x=hit_pd_c[f"{field}"]/hit_pd_c.charge_SH,
                             y=hit_pd_c.residual_tpc,
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title=f"{field} ratio")
        fig.update_yaxes(range=y_range, title="Res fit TPC [mm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, f"residuals_tpc_vs_{field}_ratio.html"), include_plotlyjs="directory")

    def plot_enemy_angle_track(self):

        x_range = [0.5, 2]
        y_range = [-0.6, 0]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Enemy_01", "Enemy_12", "Enemey_23", "--",
                                "Enemy_01", "Enemy_12", "Enemey_23", "--"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0,3):
            fig.add_trace(
                go.Histogram2d(x=(180 / np.pi) * np.arctan(self.cl_pd_x_enemy.angle_trk_x),
                               y=self.cl_pd_x_enemy[f"ene_({pl}, {pl+1})"].dropna(),
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 80},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 80},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 3):
            fig.add_trace(
                go.Box(x=(((180 / np.pi) * np.arctan(self.cl_pd_x_enemy.angle_trk_x))//0.2)*0.2,
                               y=self.cl_pd_x_enemy[f"ene_({pl}, {pl+1})"].dropna(),
                       name=f"Box det {pl}", boxpoints=False),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 3):
            fig.add_trace(
                go.Histogram(x=(180 / np.pi) * np.arctan(self.cl_pd_x_enemy.angle_trk_x),
                               y=self.cl_pd_x_enemy[f"ene_({pl}, {pl+1})"].dropna(),
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title=f"Trk angle [°]")
        fig.update_yaxes(range=y_range, title="Res enemy [cm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, f"enemy_vs_trk_angle.html"), include_plotlyjs="directory")

    def plot_enemy_angle_fit(self):
        x_range = [-10, 90]
        y_range = [-0.6, 0]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Enemy_01", "Enemy_12", "Enemey_23", "--",
                                "Enemy_01", "Enemy_12", "Enemey_23", "--"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0,3):
            fig.add_trace(
                go.Histogram2d(x=(180 / np.pi) * np.arctan(self.cl_pd_x_enemy.F1),
                               y=self.cl_pd_x_enemy[f"ene_({pl}, {pl+1})"].dropna(),
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 80},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 80},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 3):
            fig.add_trace(
                go.Box(x=(((180 / np.pi) * np.arctan(self.cl_pd_x_enemy.F1))//5)*5,
                               y=self.cl_pd_x_enemy[f"ene_({pl}, {pl+1})"].dropna(),
                       name=f"Box det {pl}", boxpoints=False),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 3):
            fig.add_trace(
                go.Histogram(x=(180 / np.pi) * np.arctan(self.cl_pd_x_enemy.F1),
                               y=self.cl_pd_x_enemy[f"ene_({pl}, {pl+1})"].dropna(),
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title=f"Fit angle [°]")
        fig.update_yaxes(range=y_range, title="Res enemy [cm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, f"enemy_vs_fit_angle.html"), include_plotlyjs="directory")


    def plot_enemy_charge(self):
        x_range = [0, 120]
        y_range = [-0.6, 0]

        fig = make_subplots(rows=4, cols=2,
                            # row_heights=[800,800,800,800],
                            subplot_titles=(
                                "Enemy_01", "Enemy_12", "Enemey_23", "--",
                                "Enemy_01", "Enemy_12", "Enemey_23", "--"
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": True}, {"secondary_y": True}],
                                [{"secondary_y": True}, {"secondary_y": True}]
                            ],
                            horizontal_spacing=0.10
                            )
        for pl in range(0,3):
            fig.add_trace(
                go.Histogram2d(x=self.cl_pd_x_enemy.cl_charge,
                               y=self.cl_pd_x_enemy[f"ene_({pl}, {pl+1})"].dropna(),
                               ybins={"start": y_range[0], "end": y_range[1],
                                      "size": (y_range[1] - y_range[0]) / 80},
                               xbins={"start": x_range[0], "end": x_range[1],
                                      "size": (x_range[1] - x_range[0]) / 80},
                               colorscale="viridis",
                               showlegend=False,
                               showscale=False, ),
                col=pl // 2 + 1, row=pl % 2 + 1)

        for pl in range(0, 3):
            fig.add_trace(
                go.Box(x = (self.cl_pd_x_enemy.cl_charge//5)*5,
                      y = self.cl_pd_x_enemy[f"ene_({pl}, {pl+1})"].dropna(),
                       name=f"Box det {pl}", boxpoints=False),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2)

        for pl in range(0, 3):
            fig.add_trace(
                go.Histogram(x=self.cl_pd_x_enemy.cl_charge,
                             y=self.cl_pd_x_enemy[f"ene_({pl}, {pl+1})"].dropna(),
                             name=f"Hist det {pl}", opacity=0.15, histnorm="percent"),
                col=pl // 2 + 1, row=pl % 2 + 1 + 2, secondary_y=True)

        fig.update_xaxes(range=x_range, title=f"Cluster charge [fC]")
        fig.update_yaxes(range=y_range, title="Res enemy [cm]", secondary_y=False)
        fig.update_yaxes(title="%", secondary_y=True)
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(self.plt_path, f"enemy_vs_charge.html"), include_plotlyjs="directory")
