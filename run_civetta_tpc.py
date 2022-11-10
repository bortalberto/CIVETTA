import os
import sys
import shutil

run = sys.argv[1]

for n in range(0, 8):
    os.mkdir(f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}")
    os.mkdir(f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/elaborated")
    os.mkdir(f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/perf")

    options = ""
    if n < 1:
        options += " -no_errors"
    if n < 2:
        options += " -no_first_last_shift"
    if n < 3:
        options += " -no_capacitive"
    if n < 4:
        options += " -drift_velocity"
    if n < 4:
        options += " -no_time_walk_corr"
    if n < 5:
        options += " -no_border_correction"
    if n < 6:
        options += " -no_prev_strip_charge_correction"
    #
    # os.system(f"./civetta_tpc.py {run} -tpc_pos {options};")
    # os.system(f"./civetta_perf.py {run} -perf -sT 20 -sD 6 -chi -cpu 20 -tpc;")
    #
    # os.system(f"./civetta_perf.py {run} -tpc -Ser;")
    #
    # os.system(f"./civetta_tpc.py {run} -angle 45 -plot_evts;")
    # os.system(f"./civetta_tpc.py {run} -angle 45 -post_plot;")

    os.system(f"{options} >> /media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/elaborated/options.txt")
    shutil.move(f"/media/disk2T/data/elaborated_output/{run}",f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/elaborated")
    shutil.move(f"/media/disk2T/data/perf_out/{run}",f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/perf")
