#!/usr/bin/env python

import planar_analysis_lib as pl_lib
from multiprocessing import Pool,cpu_count
import glob2
import os
from tqdm import tqdm
import configparser
import sys
import argparse
import tpc as tpc_lib

class runner:
    """
    This class simply manage the launch of the libs functions
    """
    def __init__(self, data_folder,run,cpu_to_use=cpu_count(), cylinder=False):
        self.data_folder = data_folder
        self.cpu_to_use = cpu_to_use
        self.run_number = run
        self.cylinder = cylinder


    def calc_and_save_thr_eff(self):
        tpc_prep = tpc_lib.tpc_prep(self.data_folder, self.cpu_to_use, self.run_number, self.cylinder)
        tpc_prep.exctract_thr_eff()

    def calc_time_and_time_walk(self):
        tpc_prep = tpc_lib.tpc_prep(self.data_folder, self.cpu_to_use, self.run_number, self.cylinder)
        tpc_prep.apply_time_walk_corr_run()
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
    except KeyError as E:
        print ("Missing or partial configration file, restore it.")
        sys.exit(1)

    if (args.data_folder):
        data_folder=args.data_folder

    if not args.Silent:
        print ("#############################################################")
        print (f"Run : {run}")
        print (f"Data_folder : {data_folder}")
        if args.thr_eff:
            print("         -Estimating thr")
        if args.time_walk:
            print("         -Calculating time and time walk")

        if args.cpu:
            print (f"Parallel on {args.cpu} CPUs")
        print ("#############################################################")

    op_list=[]
    options={}
    if args.thr_eff:
        op_list.append("thr_eff")
    if args.time_walk:
        op_list.append("time_walk")
    # if not (args.decode | args.ana | args.clusterize | args.tracking | args.selection | args.calibrate_alignment | args.compress | args.root_conv | args.performance):
    #     op_list=["D","A","C", "T","S"]

    if args.cpu:
        options["cpu_to_use"] = args.cpu
    if args.Silent:
        options["Silent"] = args.Silent

    if len (op_list)>0:
        main_runner = runner(data_folder, run, **options)
    else:
        sys.exit(0)

    if "thr_eff" in (op_list):
        main_runner.calc_and_save_thr_eff()

    if "time_walk" in (op_list):
        main_runner.calc_time_and_time_walk()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="CIVETTA perf: Tools to measure the performances of TIGER data",)
    parser.add_argument('run', type=int, help='Run number')
    parser.set_defaults(method=main)
    parser.add_argument('-df', '--data_folder', type=str, required=False,
                        help='Specify the data folder, default set by the .ini file')
    parser.add_argument('-cpu', '--cpu', help='Specify CPU count ',type=int)
    parser.add_argument('-S', '--Silent', help='Print only errors ',action="store_true")
    parser.add_argument('-thr', '--thr_eff', help='Calculate and save effective thr ', action="store_true")
    parser.add_argument('-tw', '--time_walk', help='Correct time walk ', action="store_true")

    args = parser.parse_args()
    args.method(**vars(args))
