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

warnings.filterwarnings('ignore')

def time_walk_rough_corr(charge,signal_width, a,b,c):
    charge=np.array(charge)
    if charge<=0:
        charge=0.01
    time = a/charge**b  +  c
    if time > signal_width/4+60:
        time = signal_width/4+60
    return time

def calc_tpc_pos(cluster, hits, vel_l, ref_l):
    """
    Generic function to calc the tpc position on a cluster
    :param cluster:
    :param hits:
    :param vel_l:
    :param ref_l:
    :return:
    """
    vel = vel_l[cluster.planar]
    ref_time = ref_l[cluster.planar]

    hits = hits[hits.hit_id.isin(cluster.hit_ids)]
    hits["pos_g"] = (hits.hit_time - ref_time) * vel
    if hits.strip_x.nunique() > 1:
        try:
            fit = np.polyfit(
                x=np.float64(hits.strip_x.values),
                y=np.float64(hits.pos_g.values),
                w=np.float64(1 / (hits.hit_time_error.values * vel)),
                deg=1
            )
            pos_utpc = (2.5 - fit[1]) / fit[0]
        except ValueError as E:
            print (ValueError, "Hits" , hits)
            print (f"Count: {hits['count'].mean()} ")
            pos_utpc=cluster.cl_pos_x
        return pos_utpc
    else:
        return cluster.cl_pos_x


def time_walk_rough_corr_calib(charge, a,b,c ):
    charge=np.array(charge)
    time =  a/charge**b  +  c
    return time

def calc_corr(row,dict_calibrations, thr_tmw):
    g = row.gemroc
    t = row.tiger
    c = row.channel
    return(time_walk_rough_corr(row.charge_SH, *dict_calibrations[thr_tmw[g,t,c]]))

def minus_errorfunc(x, x0, sig, c):
    y = (special.erf((x - x0) / (1.4142 * sig))) * c / 2 + 0.5 * c
    return -y


def errorfunc(x, x0, sig, c):
    y = (special.erf((x - x0) / (1.4142 * sig))) * c / 2 + 0.5 * c
    return y
def linear_func(p, x):
    """ Linear function to use like model for ODR fitting"""
    m, c = p
    return m*x + c
fit_model = Model(linear_func)
class tpc_prep:
    """
    Class to run before TPC
    """

    def __init__(self, data_folder, cpu_to_use, run, cylinder, signal_width=80, silent=False,
                 no_errors= True, no_first_last_shift=True, no_capacitive=True, drift_velocity=0, no_time_walk_corr=True,
                 no_border_correction=True, no_prev_strip_charge_correction=True):
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
        self.no_errors = no_errors
        self.no_first_last_shift = no_first_last_shift
        if self.no_first_last_shift:
            self.first_last_shift = 0
        else:
            self.first_last_shift=0.5
        self.no_capacitive = no_capacitive
        self.drift_velocity = drift_velocity
        self.no_time_walk_corr = no_time_walk_corr
        self.no_border_correction = no_border_correction
        self.no_prev_strip_charge_correction = no_prev_strip_charge_correction

    def thr_tmw(self,row):
        """
        Extract the nearest calibration value to the thr_eff
        """
        thr_poss = np.array([0.5, 1, 2, 3])
        return thr_poss[np.argmin(abs(row - thr_poss))]

    def thr_eff(self,row):
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
        hit_pd = pd.read_feather(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"hit_data-zstd.feather"))
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
            dict_calibrations[thr] = np.append(np.array([self.signal_width]), self.get_calibration_time_walk_courve(self.signal_width, thr))
        hit_pd["hit_time_corr"] = hit_pd.apply(lambda x: calc_corr(x, dict_calibrations, thr_tmw), axis=1)
        # hit_pd["hit_time"] = -(hit_pd["l1ts_min_tcoarse"] - 1567) * 6.25 - 800
        hit_pd["hit_time"] = -(hit_pd["l1ts_min_tcoarse"] - 1567) * 6.25 - hit_pd["hit_time_corr"] - 800
        hit_pd["hit_time_error"] = (  (hit_pd["hit_time_corr"]/2)**2 + 6.25/(12**1/2)**2 )**(1/2)
        return hit_pd

    def apply_time_walk_corr_run(self):
        """

        """
        hit_pd = pd.read_feather(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"hit_data-zstd.feather"))
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
        start=time.time()
        print ("Concat")
        hit_pd = pd.concat(return_list, ignore_index=True)
        print (f"Time: {time.time()-start}")
        print ("Sorting")

        hit_pd.sort_values(["subRunNo","count","hit_id"], inplace=True)
        hit_pd.reset_index(drop=True, inplace=True)
        hit_pd["hit_time_corr"] = hit_pd["hit_time_corr"].astype(np.float16)
        hit_pd["hit_time"] = hit_pd["hit_time"].astype(np.float16)
        print ("Save")
        start=time.time()

        hit_pd.to_feather(os.path.join(self.tpc_dir, f"hit_data_wt-zstd.feather"), compression='zstd')
        # hit_pd.to_pickle(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"hit_data_wt.pickle.gzip"), compression="gzip")
        print (f"Time: {time.time()-start}")


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
        ax.plot(x,y , "+",label= "data")
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
    def calc_tpc_pos_subrun(self, input):
        """
        Perform the tpc calculation on 1 subrun
        :return:
        """
        cluster_pd=input[0]
        hit_pd=input[1]
        cluster_pd_evts = cluster_pd.groupby("count")
        # hit_pd = self.hit_pd.query(f"subRunNo == {cluster_pd.subrun.mode().values[0]} and strip_x>-1")
        hit_pd["pos_g"] = np.nan
        hit_pd["dropped"] = "Not"
        hit_pd["strip_x_c"]=np.nan

        pitch = 0.650
        sx_coeff = (pitch / math.sqrt(12)) ** 2
        subrun = hit_pd.subRunNo.values[0]
        hit_pd_evts = hit_pd.groupby("count")

        for count in tqdm(cluster_pd_evts.groups, desc=f"Events subrun = {subrun}", leave=False, position=(subrun%self.cpu_to_use+1) ):
        # for count in cluster_pd_evts.groups:
            clusters = cluster_pd_evts.get_group(count)
            events_hits = hit_pd_evts.get_group(count)
            for cl_index, cluster in clusters.iterrows():

                ref_time = self.ref_time_list[cluster.planar]
                if self.drift_velocity == 0: ## Option to select the speed origin
                    vel = self.vel_list[cluster.planar]
                else:
                    vel = self.drift_velocity

                cluster_hits = events_hits[events_hits.hit_id.isin(cluster.hit_ids)]
                if self.no_time_walk_corr: ## time walk correction option
                    cluster_hits["pos_g"] = (cluster_hits.hit_time + cluster_hits.hit_time_corr - ref_time) * vel
                else:
                    cluster_hits["pos_g"] = (cluster_hits.hit_time - ref_time) * vel
                hit_pd.loc[cluster_hits.index, "pos_g_pre_cor"] = cluster_hits.pos_g
                hit_pd.loc[cluster_hits.index, "pos_g"] = cluster_hits.pos_g

                # cluster_hits = cluster_hits.query("charge_SH>0")  ## Taglia a carica >0
                cluster_hits["error_from_t"] = vel * 15 / (abs(cluster_hits.charge_SH) + 0.5)
                cluster_hits["error_from_diff"] = 0
                ## Capacitive corrections
                if not self.no_capacitive: ## Capacitive correction option
                    if cluster_hits.shape[0] > 3:
                        cluster_hits = self.check_capacitive_border_two_strips(cluster_hits, hit_pd)
                avg_charge = cluster_hits.charge_SH.sum() / cluster_hits.charge_SH.shape[0]
                if cluster_hits.shape[0] > 1:
                    cluster_hits.loc[cluster_hits.index, "previous_strip_charge"] = cluster_hits.charge_SH.shift()
                    cluster_hits["charge_ratio_p"] = cluster_hits["charge_SH"] / cluster_hits["previous_strip_charge"]
                    if not self.no_prev_strip_charge_correction:
                        cluster_hits.loc[cluster_hits["charge_ratio_p"] < 1, "pos_g"] = cluster_hits["pos_g"] + 1.3 - 1.3 * cluster_hits["charge_ratio_p"]
                        hit_pd.loc[cluster_hits.index, "pos_g"] = cluster_hits.pos_g

                    if not self.no_errors:
                        error_x = np.sqrt(sx_coeff + sx_coeff * (avg_charge / cluster_hits.charge_SH))
                        error_y = cluster_hits.error_from_t.values
                    else:
                        error_x = cluster_hits.error_from_t.values*0 + 1
                        error_y = cluster_hits.error_from_t.values*0 + 1
                    pos_x_fit = np.float64(cluster_hits.strip_x.values) * pitch
                    pos_x_fit[0] = pos_x_fit[0] + self.first_last_shift
                    pos_x_fit[-1] = pos_x_fit[-1] - self.first_last_shift
                    hit_pd.loc[cluster_hits.index, "strip_x_c"][0] = pos_x_fit

                    data = RealData(pos_x_fit,
                                    np.float64(cluster_hits.pos_g.values),
                                    sx = error_x,
                                    sy = error_y)

                    odr = ODR(data, fit_model, beta0=[0.5, -20])
                    out = odr.run()
                    fit = out.beta

                    hit_pd.loc[cluster_hits.index, "error_x"] = error_x
                    hit_pd.loc[cluster_hits.index, "error_y"] = error_y
                    hit_pd.loc[cluster_hits.index, "residual_tpc"] = cluster_hits.pos_g - cluster_hits.strip_x * pitch * fit[0] - fit[1]
                    hit_pd.loc[cluster_hits.index, "strip_min"] = cluster_hits.strip_x.values[0]
                    hit_pd.loc[cluster_hits.index, "strip_max"] = cluster_hits.strip_x.values[-1]
                    hit_pd.loc[cluster_hits.index, "next_strip_charge"] = cluster_hits.charge_SH.shift(-1)
                    hit_pd.loc[cluster_hits.index, "previous_strip_charge"] = cluster_hits.charge_SH.shift()
                    hit_pd.loc[cluster_hits.index, "total_charge"] = cluster_hits.charge_SH.sum()
                    hit_pd.loc[cluster_hits.index, "cl_size"] = cluster_hits.charge_SH.shape[0]
                    hit_pd.loc[cluster_hits.index, "residual_tpc_sum"] = np.sum(cluster_hits.pos_g.values - cluster_hits.strip_x.values * pitch * fit[0] - fit[1])
                    cluster_pd.loc[cl_index, "pos_tpc"] = ((2.5 - fit[1]) / fit[0])/0.650
                    cluster_pd.loc[cl_index, "F0"] = fit[1]
                    cluster_pd.loc[cl_index, "F1"] = fit[0]

                    #         hit_pd.loc[]

                    ## Tagliando i bordi

        return hit_pd, cluster_pd

    def calc_tpc_pos(self, cpus=30):
        hit_pd = pd.read_feather(os.path.join(self.tpc_dir, f"hit_data_wt-zstd.feather"))
        hit_pd = hit_pd.sort_values(["subRunNo", "count", "planar", "strip_x"]).reset_index(drop=True) ## Sorting the values for later use
        self.vel_list = []
        self.ref_time_list = []
        if not self.silent:
            print ("\n Calculation drift velocity and reference time")
        ### Calc ref time and vel
        for pl in tqdm(range (0,4), desc="Planar", leave=False):
            cluster_pd_eff_cc = pd.read_feather(os.path.join(self.data_folder,"perf_out", f"{self.run_number}",f"match_cl_{pl}-zstd.feather"))
            ## Selecting only hit inside clusters (only in X)
            cluster_pd_eff_cc = cluster_pd_eff_cc.query(f"cl_pos_x>-2")
            hit_pd_c = hit_pd.query(f"planar=={pl}")
            hit_pd_c = hit_pd_c.query(f"strip_x>-1")
            total_mask = hit_pd_c.charge_SH > 1000
            for subrun in tqdm(cluster_pd_eff_cc.subrun.unique(), leave = False):
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
            self.ref_time_list.append(ref_time)
        cluster_pd = pd.read_feather(os.path.join(self.data_folder, "raw_root",f"{self.run_number}", "cluster_pd_1D-zstd.feather"))
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
            print ("Clusters 2-D")
        if len(sub_list) > 0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(sub_list), disable=self.silent) as pbar_2:
                    for i, x in enumerate(pool.imap_unordered(self.build_2D_clusters, sub_list)):
                        pd_2d_return_list.append(x)
                        pbar_2.update()
        print ("Concatenating")
        cluster_pd_2D = pd.concat(pd_2d_return_list)
        print ("Saving")
        cluster_pd_2D.reset_index(inplace=True, drop=True)
        cluster_pd_2D.to_feather(os.path.join(self.tpc_dir, f"cluster_pd_2D_TPC-zstd.feather"), compression='zstd')

    def build_2D_clusters(self, cluster_pd):
        dict_4_pd = {
            "run"        : [],
            "subrun"     : [],
            "count"      : [],
            "planar"     : [],
            "cl_pos_x"   : [],
            "cl_pos_x_tpc": [],
            "cl_pos_y"   : [],
            "cl_charge"  : [],
            "cl_charge_x": [],
            "cl_charge_y": [],
            "cl_size_x"  : [],
            "cl_size_y"  : [],
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

# class out_plots_class:
#     """
#     Post analysis class to assert the scan performances
#     """
#     def __init__(self):
#         self.events_pd_hit=
#         self.eff_pd=