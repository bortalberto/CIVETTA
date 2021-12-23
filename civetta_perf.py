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

class runner:
    """
    This class simply manage the launch of the libs functions
    """
    def __init__(self, data_folder,run,cpu_to_use=cpu_count(), cylinder=False, sigmas_trackers=1, sigmas_DUT=5 ):
        self.data_folder = data_folder
        self.cpu_to_use = cpu_to_use
        self.run_number = run
        self.cylinder = cylinder

    def eval_perf(self,put):
        perf.calculte_eff(self.run_number, self.data_folder, put, self.cpu_to_use)

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
        if args.cpu:
            print (f"Parallel on {args.cpu} CPUs")
        print ("#############################################################")

    op_list=[]
    if args.performance:
        if args.performance in (0,1,2,3,-1):
            op_list.append("perf")
        else:
            print ("Bad argument for performance option. Use the planar number [0..3] or -1 to run on all")

    # if not (args.decode | args.ana | args.clusterize | args.tracking | args.selection | args.calibrate_alignment | args.compress | args.root_conv | args.performance):
    #     op_list=["D","A","C", "T","S"]

    options={}
    if args.cpu:
        options["cpu_to_use"]=args.cpu
    if args.Silent:
        options["Silent"]=args.Silent
    if len (op_list)>0:
        main_runner = runner(data_folder,run,calib_folder,mapping_file,**options)
    else:
        sys.exit(0)


    if "perf" in (op_list):
        main_runner.eval_perf(args.performance)



if __name__=="__main__":

    parser = argparse.ArgumentParser(description="CIVETTA perf: Tools to measure the performances of TIGER data",)
    parser.add_argument('run', type=int, help='Run number')
    parser.set_defaults(method=main)
    parser.add_argument('-df','--data_folder', type=str, required=False,
                        help='Specify the data folder, default set by the .ini file')
    parser.add_argument('-cpu','--cpu', help='Specify CPU count ',type=int)
    parser.add_argument('-S','--Silent', help='Print only errors ',action="store_true")
    parser.add_argument('-sT','--sigmas_trackers', help='Sigma trackers', type=int, default=1)
    parser.add_argument('-chi','--chi_sqared', help='Use chi squared cut on trackers',action="store_true")
    parser.add_argument('-sD','--sigmas_DUT', help='Sigma DUT', type=int, default=5)


    args = parser.parse_args()
    args.method(**vars(args))