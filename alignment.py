import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import sys
sys.path.append('../analisi_planari')
import plotly.graph_objects as go

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
    """

    :param subrun_tgt:
    :return:
    """
    run_l = []
    subrun_l = []
    count_l = []
    x_fit = []
    y_fit = []
    planar_di = {
        "res_planar_0_x": [],
        "res_planar_1_x": [],
        "res_planar_2_x": [],
        "res_planar_3_x": [],
        "res_planar_0_y": [],
        "res_planar_1_y": [],
        "res_planar_2_y": [],
        "res_planar_3_y": []

    }
    cl_id_l=[]

    for run in tqdm(cluster_pd_1D["run"].unique(), desc= "Run", leave =None):
        cluster_pd_1D_c0 = cluster_pd_1D[cluster_pd_1D.run == run]
        for subrun in tqdm(cluster_pd_1D_c0["subrun"].unique(), desc="Subrun", leave=None):
            data_pd_cut_1 = cluster_pd_1D_c0[cluster_pd_1D_c0.subrun == subrun]
            for count in data_pd_cut_1["count"].unique():
                df_c2 = data_pd_cut_1[data_pd_cut_1["count"] == count] # df_c2 is shorter

                # Build track X
                if len(df_c2[df_c2.cl_pos_x_cm>0].planar.unique())>2: ## I want at least 3 point in that view
                    fit_x, cl_ids, res_dict = fit_tracks_view(df_c2[df_c2.cl_pos_x_cm>0], "x")
                    run_l.append(run)
                    subrun_l.append(subrun)
                    count_l.append(count)
                    x_fit.append(fit_x)
                    y_fit.append(np.nan)
                    for planar in range(0,4):
                        if planar in res_dict.keys():
                            planar_di[f"res_planar_{planar}_x"].append(res_dict[planar])
                            planar_di[f"res_planar_{planar}_y"].append(np.nan)
                        else:
                            planar_di[f"res_planar_{planar}_x"].append(np.nan)
                            planar_di[f"res_planar_{planar}_y"].append(np.nan)
                    cl_id_l.append(cl_ids)
                # Build track Y
                if len(df_c2[df_c2.cl_pos_y_cm>0].planar.unique())>2: ## I want at least 3 point in that view
                    fit_y, cl_ids,res_dict = fit_tracks_view(df_c2[df_c2.cl_pos_y_cm>0], "y")
                    run_l.append(run)
                    subrun_l.append(subrun)
                    count_l.append(count)
                    x_fit.append(np.nan)
                    y_fit.append(fit_y)
                    for planar in range(0, 4):
                        if planar in res_dict.keys():
                            planar_di[f"res_planar_{planar}_x"].append(np.nan)
                            planar_di[f"res_planar_{planar}_y"].append(res_dict[planar])
                        else:
                            planar_di[f"res_planar_{planar}_x"].append(np.nan)
                            planar_di[f"res_planar_{planar}_y"].append(np.nan)
                    cl_id_l.append(cl_ids)


    dict_4_pd = {
        "run": run_l,
        "subrun": subrun_l,
        "count": count_l,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "res_planar_0_x": planar_di["res_planar_0_x"],
        "res_planar_1_x": planar_di["res_planar_1_x"],
        "res_planar_2_x": planar_di["res_planar_2_x"],
        "res_planar_3_x": planar_di["res_planar_3_x"],
        "res_planar_0_y": planar_di["res_planar_0_y"],
        "res_planar_1_y": planar_di["res_planar_1_y"],
        "res_planar_2_y": planar_di["res_planar_2_y"],
        "res_planar_3_y": planar_di["res_planar_3_y"],
        "cl_ids":cl_id_l
    }
    return ( pd.DataFrame(dict_4_pd) )

def fit_tracks_view( df, view):
    """
    Builds tracks on 1 view
    :param df:
    :return:
    """
    pd_fit_l = [] ## list of rows to fit
    ids = []
    for planar in df.planar.unique():
        df_p=df[df.planar==planar] ## select planar
        to_fit = df_p[df_p['cl_charge'] == df_p['cl_charge'].max()] ## Finds maximum charge cluster

        if len (to_fit)>1: ## If we are 2 cluster with the exact same charge...
            pd_fit_l.append(to_fit.iloc[0])
            ids.append(to_fit.iloc[0].cl_id.values[0])

        else:
            pd_fit_l.append(to_fit)
            ids.append((planar,to_fit.cl_id.values[0]))

    pd_fit=pd.concat(pd_fit_l)

    fit = fit_1_d(pd_fit.cl_pos_z_cm, pd_fit[f"cl_pos_{view}_cm"])
    res_dict={}
    for planar in df.planar.unique():
        pd_fit_pl=pd_fit[pd_fit.planar==planar]
        res_dict[planar]=calc_res(pd_fit_pl[f"cl_pos_{view}_cm"], fit, pd_fit_pl.cl_pos_z_cm)
    return fit, ids, res_dict


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