import numpy as np
import pandas as pd
import os
import requests
from scipy.optimize import curve_fit
from multiprocessing import Pool,cpu_count
from tqdm import tqdm
from itertools import chain
import time
def time_walk_rough_corr(charge,signal_width, a,b,c):
    charge=np.array(charge)
    if charge<=0:
        charge=0.01
    time = a/charge**b  +  c
    if time > signal_width/4+60:
        time = signal_width/4+60
    return time


def time_walk_rough_corr_calib(charge, a,b,c ):
    charge=np.array(charge)
    time =  a/charge**b  +  c
    return time

def calc_corr(row,dict_calibrations, thr_tmw):
    g = row.gemroc
    t = row.tiger
    c = row.channel
    return(time_walk_rough_corr(row.charge_SH, *dict_calibrations[thr_tmw[g,t,c]]))


class tpc_prep:
    """
    Class to run before TPC
    """

    def __init__(self, data_folder, cpu_to_use, run, cylinder, signal_width=80):
        self.data_folder = data_folder
        self.cpu_to_use = cpu_to_use
        self.run_number = run
        self.cylinder = cylinder
        self.signal_width =signal_width

    def thr_tmw(self,row):
        """
        Extract the nearest calibration value to the the thr_eff
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
            print("Calibration ot found")
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
        hit_pd = pd.read_pickle(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"hit_data.pickle.gzip"), compression="gzip")
        hit_pd = hit_pd.query("charge_SH<10")
        thr_eff = hit_pd.groupby(["gemroc", "tiger", "channel"]).charge_SH.agg(self.thr_eff)
        thr_tmw = thr_eff.apply(self.thr_tmw)
        thr_eff.to_pickle(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"thr_eff.pickle"))
        thr_tmw.to_pickle(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"thr_tmw.pickle"))


    def apply_time_walk_corr_subrun(self, hit_pd):
        """

        """
        thr_tmw = pd.read_pickle("/media/disk2T/VM_work_zone/data/raw_root/563/thr_tmw.pickle.gzip")
        dict_calibrations = {}
        for thr in (0.5, 1, 2, 3):
            dict_calibrations[thr] = np.append(np.array([self.signal_width]), self.get_calibration_time_walk_courve(self.signal_width, thr))
        hit_pd["hit_time_corr"] = hit_pd.apply(lambda x: calc_corr(x, dict_calibrations, thr_tmw), axis=1)
        hit_pd["hit_time"] = -(hit_pd["l1ts_min_tcoarse"] - 1567) * 6.25 - hit_pd["hit_time_corr"] - 800
        hit_pd["hit_time_error"] = (64 + (hit_pd["hit_time_corr"]/5)**2)**(1/2)
        return hit_pd

    def apply_time_walk_corr_run(self):
        """

        """
        hit_pd = pd.read_pickle(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"hit_data.pickle.gzip"), compression="gzip")
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
        print (f"Time: {start-time.time()}")
        print ("Sorting")

        hit_pd.sort_values("hit_id", inplace=True)
        hit_pd.reset_index(drop=True, inplace=True)
        print ("Save")

        hit_pd.to_pickle(os.path.join(self.data_folder, "raw_root", f"{self.run_number}", f"hit_data.pickle.gzip"), compression="gzip")