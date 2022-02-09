#!/usr/bin/env python

import planar_analysis_lib as pl_lib
from multiprocessing import Pool,cpu_count
import glob2
import os
from tqdm import tqdm
import configparser
import sys
import argparse
import pandas as pd
import json
import numpy as np
import perf
import perf_out_evt_display as perfo

class runner:
    """
    This class simply manage the launch of the libs functions
    """
    def __init__(self, data_folder,run,cpu_to_use=cpu_count(), cylinder=False, sigmas_trackers=1, sigmas_DUT=5, chi_sqared=False, multi_tracks_suppresion=False, hit_efficiency=False):
        self.data_folder = data_folder
        self.cpu_to_use = cpu_to_use
        self.run_number = run
        self.cylinder = cylinder
        self.sigmas_trackers = sigmas_trackers
        self.sigmas_DUT = sigmas_DUT
        self.chi_squared = chi_sqared
        self.multi_tracks_suppresion=multi_tracks_suppresion
        self.hit_efficiency = hit_efficiency

    def eval_perf(self,put):
        print (f"Sigmas trackers: {self.sigmas_trackers}, sigmas DUT: {self.sigmas_DUT}")

        perf.calculte_eff(self.run_number, self.data_folder, put, self.cpu_to_use,
                        nsigma_put=self.sigmas_DUT, nsigma_trackers=self.sigmas_trackers, chi_sq_trackers=self.chi_sq_trackers, multi_tracks_suppresion=self.multi_tracks_suppresion, hit_efficiency=self.hit_efficiency)

    def save_events(self, put, nevents):
        perfo.save_evt_display(self.run_number, self.data_folder, put, nevents)

    def save_eff(self, put):
        if put in (0,1,2,3):
            planar_list = [put,]
        else:
            planar_list=[0,1,2,3]
        perfo.extract_eff_and_res(self.run_number, self.data_folder, planar_list)

##############################################################################################
##																							##
##										MAIN												##
##																							##
##############################################################################################
def main(run, **kwargs):
    """
    That main programm can be called using the console or by another Python program importing this file
    :param run:
    :param kwargs:
    :return:
    """

    config=configparser.ConfigParser()
    config_file="config.ini"

    config.read(os.path.join(sys.path[0], config_file))
    try:
        data_folder=config["GLOBAL"].get("data_folder")
        if data_folder=="TER":
            data_folder=os.environ["TER_data"]
        mapping_file=config["GLOBAL"].get("mapping_file")
        calib_folder=config["GLOBAL"].get("calib_folder")
    except KeyError as E:
        print ("Missing or partial configration file, restore it.")
        sys.exit(1)

    if (args.data_folder):
        data_folder=args.data_folder

    if not args.Silent:
        print ("#############################################################")
        print (f"Run : {run}")
        print (f"Data_folder : {data_folder}")
        if args.performance:
            print("         -Performance evaluation")
        if args.save_eff:
            print("         -Calculating and saving eff")
        if args.save_events:
            print("         -Saving events")
        if args.cpu:
            print (f"Parallel on {args.cpu} CPUs")
        print ("#############################################################")

    op_list=[]
    options={}
    if args.performance:
        op_list.append("perf")
        options["sigmas_trackers"] = args.sigmas_trackers
        options["sigmas_DUT"] = args.sigmas_DUT
        if args.multi_tracks_suppresion:
            print("Using multi_tracks_suppresion \n")
        options["multi_tracks_suppresion"] = True

    # if not (args.decode | args.ana | args.clusterize | args.tracking | args.selection | args.calibrate_alignment | args.compress | args.root_conv | args.performance):
    #     op_list=["D","A","C", "T","S"]

    if args.cpu:
        options["cpu_to_use"] = args.cpu
    if args.Silent:
        options["Silent"] = args.Silent
    if args.hit_efficiency:
        options["hit_efficiency"] = args.hit_efficiency
        print ("Efficiency on hits")
    if args.chi_squared:
        options["chi_squared"] = True
    if args.save_events:
        op_list.append("save_events")
    if args.save_eff:
        op_list.append("save_eff")

    if len (op_list)>0:
        main_runner = runner(data_folder, run, **options)
    else:
        sys.exit(0)

    if args.Detector_under_test == -1:
        args.Detector_under_test = [0,1,2,3]

    if "perf" in (op_list):
        main_runner.eval_perf(args.Detector_under_test)
    if "save_eff" in (op_list):
        main_runner.save_eff(args.Detector_under_test)
    if "save_events" in (op_list):
        main_runner.save_events(args.save_events[0], args.save_events[1])


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="CIVETTA perf: Tools to measure the performances of TIGER data",)
    parser.add_argument('run', type=int, help='Run number')
    parser.set_defaults(method=main)
    parser.add_argument('-df','--data_folder', type=str, required=False,
                        help='Specify the data folder, default set by the .ini file')
    parser.add_argument('-cpu','--cpu', help='Specify CPU count ',type=int)
    parser.add_argument('-S','--Silent', help='Print only errors ',action="store_true")
    parser.add_argument('-sT','--sigmas_trackers', help='Sigma trackers', type=float, default=1)
    parser.add_argument('-chi','--chi_squared', help='Use chi squared cut on trackers',action="store_true")
    parser.add_argument('-sD','--sigmas_DUT', help='Sigma DUT', type=int, default=5)
    parser.add_argument('-perf','--performance', help='Performance evaluation ', action="store_true")
    parser.add_argument('-mt','--multi_tracks_suppresion', help='Activate suppression of multi tracks events',action="store_true")
    parser.add_argument('-ht','--hit_efficiency', help='Calculate efficiency with hits',action="store_true")

    ## Options about saving the output
    parser.add_argument('-Se','--save_events', help='Save the html file of some events, specify the planar and the number of events',type=int, nargs=2)
    parser.add_argument('-Ser','--save_eff', help='Calculate the efficiency and the noise contribution, the resolution and the enemy resolution',action="store_true")

    parser.add_argument('-Dut','--Detector_under_test', help='Specify one detector under test',type=int, default=-1)

    args = parser.parse_args()
    args.method(**vars(args))
