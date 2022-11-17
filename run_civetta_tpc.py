import os
import sys
import shutil

run = sys.argv[1]
angle = sys.argv[2]
if not os.path.isdir(f"/media/disk2T/VM_work_zone/data/an_scan/{run}"):
    os.mkdir(f"/media/disk2T/VM_work_zone/data/an_scan/{run}")
for n in range(0, 8):
    if not os.path.isdir(f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}"):

        os.mkdir(f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}")

    options = ""
    if n < 1:
        options += " -no_errors"
    if n < 2:
        options += " -no_first_last_shift"
    if n < 3:
        options += " -no_capacitive"
    if n < 4:
        options += " -no_time_walk_corr"
    if n < 5:
        options += " -no_border_correction"
    if n < 6:
        options += " -no_prev_strip_charge_correction"
    if n <7:
        options += " -no_pos_g_cut"
    print (options)
    with open(f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/options.txt", "w+") as option_file:
        option_file.write(options)


    os.system(f"./civetta_tpc.py {run} -tpc_angle {angle} -tpc_pos {options} ;")
    os.system(f"./civetta_perf.py {run} -perf -sT 20 -sD 6 -chi -cpu 20 -tpc;")

    os.system(f"./civetta_perf.py {run} -tpc -Ser;")

    os.system(f"./civetta_tpc.py {run} -angle {angle} -plot_evts -tpc_angle {angle};")
    os.system(f"./civetta_tpc.py {run} -angle {angle} -post_plot -tpc_angle {angle};")

    shutil.move(f"/media/disk2T/VM_work_zone/data/elaborated_output/{run}/",f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/")
    os.rename(f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/{run}", f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/elaborated_output")
    shutil.copytree(f"/media/disk2T/VM_work_zone/data/perf_out/{run}/",f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/perf")
    # os.rename(f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/{run}", f"/media/disk2T/VM_work_zone/data/an_scan/{run}/{n}/perf_out")
