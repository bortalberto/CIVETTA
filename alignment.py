import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import sys
sys.path.append('../analisi_planari')
import plotly.graph_objects as go
from multiprocessing import Pool,cpu_count
import configparser
import os
from planar_analysis_lib import calc_res,fit_1_d
from planar_analysis_lib import tracking_1d
config=configparser.ConfigParser()
config.read(os.path.join(sys.path[0], "config.ini"))
try:
    data_folder = config["GLOBAL"].get("data_folder")

except KeyError as E:
    print (f"{E}Missing or partial configration file, restore it.")
    sys.exit(1)

if data_folder=="TER":
    try:
        data_folder=os.environ["TER_data"]
    except KeyError as E:
        print(f"{E} is not defined in your system variables")
        sys.exit(1)

def build_tracks_pd( cluster_pd_1D):
    tracking_return_list = []
    tracker = tracking_1d(0, data_folder)
    for run in tqdm(cluster_pd_1D.run.unique(), desc="Run", leave=None):
        tracker.cluster_pd_1D=cluster_pd_1D[cluster_pd_1D.run==run]
        subrun_list = (tracker.read_subruns())
        with Pool(processes=cpu_count()) as pool:
            with tqdm(total=len(subrun_list), desc="Subrun", leave=None) as pbar:
                for i, x in enumerate(pool.imap_unordered(tracker.build_tracks_pd, subrun_list)):
                    tracking_return_list.append(x)
                    pbar.update()
        tracker.tracks_pd = pd.concat(tracking_return_list)
    return tracker.tracks_pd


def build_displacement_pd(track_pd):
    """
    Calculate the planar displacement
    :param track_pd:
    :return:
    """
    return_dict = {}
    for planar in [0, 1, 2, 3]:
        for view in ["x", "y"]:
            return_dict[f"displ_planar_{planar}_{view}"] = np.mean(track_pd[f"res_planar_{planar}_{view}"])
    return return_dict

def align_runs(run_list, iterations=1):
    """
    Aligns the run in the list, each one will have the same correction array
    :param run_list:
    :return:
    """
    c_pd_list = []
    t_pd_list = []
    for run_number in run_list:
        # loads all the data in the macrorun
        c_pd_list.append(pd.read_pickle("{}/raw_root/{}/cluster_pd_1D.pickle.gzip".format(data_folder, run_number), compression="gzip"))
        t_pd_list.append(pd.read_pickle("{}/raw_root/{}/tracks_pd_1D.pickle.gzip".format(data_folder, run_number), compression="gzip"))
    cluster_pd = pd.concat(c_pd_list)
    track_pd = pd.concat(t_pd_list)

    # Add infos about the cluster position (to be fixed when adding angle)
    cluster_pd["cl_pos_x_cm"] = cluster_pd.cl_pos_x * 0.0650
    cluster_pd["cl_pos_y_cm"] = cluster_pd.cl_pos_y * 0.0650
    cluster_pd["cl_pos_z_cm"] = cluster_pd.planar * 10

    # Initialize the dict for correction and displacement
    displ=build_displacement_pd(track_pd)
    correction = {
        0: {"x": 0, "y": 0},
        1: {"x": 0, "y": 0},
        2: {"x": 0, "y": 0},
        3: {"x": 0, "y": 0}

    }

    # Performs N rounds of alignment
    for j in tqdm(range(0, iterations), desc="Iterations"):
        for planar in tqdm((0, 1, 2, 3), desc= "Planar"):
            for view in ("x", "y"):
                cluster_pd.loc[cluster_pd.planar == planar, f"cl_pos_{view}_cm"] = cluster_pd.loc[cluster_pd.planar == planar, f"cl_pos_{view}_cm"] - displ[f"displ_planar_{planar}_{view}"]
                correction[planar][view] += displ[f"displ_planar_{planar}_{view}"]
            corr_tracks = build_tracks_pd(cluster_pd)
            displ = build_displacement_pd(corr_tracks)
    for run in run_list:
        pickle.dump(correction, open(os.path.join(data_folder, "alignment", f"{run}"), 'wb'))
    return corr_tracks, track_pd

def apply_correction(cluster_pd, corr):
    cluster_pd["cl_pos_x_cm"] = cluster_pd.cl_pos_x * 0.0650
    cluster_pd["cl_pos_y_cm"] = cluster_pd.cl_pos_y * 0.0650
    cluster_pd["cl_pos_z_cm"] = cluster_pd.planar * 10
    for planar in (0,1,2,3):
        for view in ("x","y"):
            cluster_pd.loc[cluster_pd.planar==planar, f"cl_pos_{view}_cm"]=cluster_pd.loc[cluster_pd.planar==planar, f"cl_pos_{view}_cm"]-corr[planar][view]
    return cluster_pd

def display_results(corr_tracks, track_pd):
    fig_list = []
    for planar in (0, 1, 2, 3):
        for view in ("x", "y"):
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=track_pd[f"res_planar_{planar}_{view}"], nbinsx=1000, xbins=dict(  # bins used for histogram
                start=-0.2,
                end=0.2
            ), name="Pre allineamneto"))
            fig.add_trace(go.Histogram(x=corr_tracks[f"res_planar_{planar}_{view}"], nbinsx=1000, xbins=dict(  # bins used for histogram
                start=-0.2,
                end=0.2),
                                       name="Post allineamneto"))
            fig.update_layout(barmode='overlay')
            fig.update_traces(opacity=0.50)
            fig_list.append(fig)
    for fig in fig_list:
        fig.show()
if __name__ == "__main__":
    corr,old = align_runs((29,),1)
    display_results(corr, old)