#!/usr/bin/env python

from multiprocessing import cpu_count
import os
import configparser
import sys
import argparse
import tpc as tpc_lib

class runner:
    """
    This class simply manage the launch of the libs functions
    """
    def __init__(self, data_folder,run,cpu_to_use=cpu_count(), cylinder=False, Silent=False):
        self.data_folder = data_folder
        self.cpu_to_use = cpu_to_use
        self.run_number = run
        self.cylinder = cylinder
        self.silent = Silent
        self.tpc_opt = []
        self.tpc_prep = tpc_lib.tpc_prep(self.data_folder, self.cpu_to_use, self.run_number, self.cylinder, silent=self.silent)

        self.tpc_prep.no_error = False
        self.tpc_prep.no_first_last_shift = False
        self.tpc_prep.no_capacitive = False
        self.tpc_prep.drift_velocity = 0
        self.tpc_prep.no_time_walk_corr = False
        self.tpc_prep.no_border_correction = False
        self.tpc_prep.no_prev_strip_charge_correction = False


    def calc_and_save_thr_eff(self):
        self.tpc_prep.exctract_thr_eff()

    def calc_time_and_time_walk(self):
        self.tpc_prep.apply_time_walk_corr_run()

    def tpc_position_clusters(self):
        self.tpc_prep.calc_tpc_pos(cpus=34)

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
    config_file="TPC_config.ini"
    config.read(os.path.join(sys.path[0], config_file))
    print (config.items())
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
        if args.tpc_position_clusters:
            print("         -Calculating TPC position on clusters")
        if args.cpu:
            print (f"Parallel on {args.cpu} CPUs")
        print ("#############################################################")

    op_list=[]
    options={}
    if args.thr_eff:
        op_list.append("thr_eff")
    if args.time_walk:
        op_list.append("time_walk")
    if args.tpc_position_clusters:
        op_list.append("tpc_position_clusters")
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
    ### TPC options
    # Default from ini
    main_runner.no_errors = config["TPC"].get("no_errors")
    main_runner.no_first_last_shift = config["TPC"].get("no_first_last_shift")
    main_runner.no_capacitive = config["TPC"].get("no_capacitive")
    main_runner.drift_velocity = config["TPC"].get("drift_velocity")
    main_runner.no_time_walk_corr = config["TPC"].get("no_time_walk_corr")
    main_runner.no_border_correction = config["TPC"].get("no_border_correction")
    main_runner.no_prev_strip_charge_correction = config["TPC"].get("no_prev_strip_charge_correction")
    # Changes from options
    if args.no_errors:
        main_runner.tpc_prep.no_errors = args.no_errors
    if args.no_first_last_shift:
        main_runner.tpc_prep.no_first_last_shift = args.no_first_last_shift
    if args.no_capacitive:
        main_runner.tpc_prep.no_capacitive = args.no_capacitive
    if args.drift_velocity:
        main_runner.tpc_prep.drift_velocity = args.drift_velocity
    if args.no_time_walk_corr:
        main_runner.tpc_prep.no_time_walk_corr = args.no_time_walk_corr
    if args.no_border_correction:
        main_runner.tpc_prep.no_border_correction = args.no_border_correction
    if args.no_prev_strip_charge_correction:
        main_runner.tpc_prep.no_prev_strip_charge_correction = args.no_prev_strip_charge_correction
    if "thr_eff" in (op_list):
        main_runner.calc_and_save_thr_eff()

    if "time_walk" in (op_list):
        main_runner.calc_time_and_time_walk()
    if "tpc_position_clusters" in (op_list):
        main_runner.tpc_position_clusters()
    # main_runner.tpc_opt = args.__dict__




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
    parser.add_argument('-tpc_pos', '--tpc_position_clusters', help='Calculate TPC position on all clusters ', action="store_true")
    parser.add_argument('-no_errors', help="Use errors or not in TPC", action="store_true")
    parser.add_argument('-no_first_last_shift', help = "Shifts first and last strip toward the center", action="store_true")
    parser.add_argument('-no_capacitive', help="Use capacitive corrections", action = "store_true")
    parser.add_argument('-drift_velocity', help="Value to be used for drift velocity", type=float)
    parser.add_argument('-no_time_walk_corr', help="Use time walk correction", action = "store_true")
    parser.add_argument('-no_border_correction', help="Use border_correction correction", action = "store_true")
    parser.add_argument('-no_prev_strip_charge_correction', help="Use correction from previous strip charge", action = "store_true")

    args = parser.parse_args()

    args.method(**vars(args))