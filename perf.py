import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import time
from multiprocessing import Pool
import pickle
import glob
import scipy.integrate
import  sys
import configparser
import root_fit_lib as r_fit

def get_run_data(runs, dtype="h", data_folder=""):
    """
    Generic functions to load data
    :param runs:
    :param dtype:
    :param data_folder:
    :return:
    """
    if dtype=="h":
        filename="hit_data"
    if dtype=="t":
        filename="tracks_pd_1D"
    if dtype=="ta":
        filename="tracks_pd_1D_align"
    if dtype=="s":
        filename="sel_cluster_pd_1D"
    if dtype=="1D":
        filename="cluster_pd_1D"
    if dtype=="2D":
        filename="cluster_pd_2D"

    data_list=[]
    for run in runs:
        data_list.append(pd.read_pickle(f"{data_folder}/raw_root/{run}/{filename}.pickle.gzip", compression="gzip"))

    return pd.concat(data_list)

def load_cluster_2D_align(runs, data_folder):
    #Load cluster data
    cl_pd_2D=get_run_data([runs],'2D', data_folder)
#     cl_pd_2D=cl_pd_2D[0:100000]
    #Calculate standard position
    cl_pd_2D["cl_pos_x_cm"] = cl_pd_2D.cl_pos_x * 0.0650
    cl_pd_2D["cl_pos_y_cm"] = cl_pd_2D.cl_pos_y * 0.0650
    cl_pd_2D["cl_pos_z_cm"] = cl_pd_2D.planar * 10
    #Drop old position to save memory
    cl_pd_2D=cl_pd_2D.drop(columns=["cl_pos_x","cl_pos_y"])
    #Drop charge and size position, not needed for alinment
    cl_pd_2D=cl_pd_2D.drop(columns=["cl_charge","cl_charge_x","cl_charge_y","cl_size_x","cl_size_y","cl_size_tot"])
    #Let's keep only events with 4 planars
#     cl_pd_2D=cl_pd_2D.groupby(["subrun","count"]).filter(lambda x: set(x["planar"])=={0,1,2,3})
    return cl_pd_2D


def fit_tracks_manager(cl_pd, planar="None", tracking_fit=False):
    """
    Manages the parallelizing
    """
    sub_data = cl_pd.groupby(["run", "subrun"])
    sub_list = []
    return_list = []
    for key in sub_data.groups:
        sub_list.append(sub_data.get_group(key))
    if len(sub_list) > 0:
        with Pool(processes=20) as pool:
            with tqdm(total=len(sub_list), desc="Tracks fitting", leave=False) as pbar:
                for i, x in enumerate(pool.imap_unordered(fit_tracks_process_pd(planar, tracking_fit), sub_list)):
                    return_list.append(x)
                    pbar.update()
        track_pd = pd.concat(return_list)
    track_pd = track_pd.reset_index()
    track_pd = track_pd.drop(columns="level_1")

    return track_pd


class fit_tracks_process_pd(object):
    def __init__(self, planar, tracking_fit):
        self.put = planar
        self.tracking_fit = tracking_fit

    def __call__(self, cl_pd):
        cl_pd = cl_pd.reset_index()
        if self.put == "None":
            tracks_pd = cl_pd.groupby(["count"])[["run", "subrun", "cl_pos_x_cm", "cl_pos_y_cm", "cl_pos_z_cm", "planar"]].apply(fit_tracks_process_row)
        else:
            tracks_pd = cl_pd.groupby(["count"])[["run", "subrun", "cl_pos_x_cm", "cl_pos_y_cm", "cl_pos_z_cm", "planar"]].apply(lambda x: fit_tracks_process_row(x, self.put, self.tracking_fit))
        return tracks_pd


def fit_tracks_process_row(x, put="None", tracking_fit=False):
    x = x.sort_values("planar")
    if tracking_fit:
        x = x[x.planar != put]

    fit_x = np.polyfit(x[(x.planar != put) & (x.cl_pos_x_cm.notna())]["cl_pos_z_cm"].dropna(), x[x.planar != put]["cl_pos_x_cm"].dropna(), 1)
    pos_x = fit_x[1] + fit_x[0] * x["cl_pos_z_cm"].values
    res_x = fit_x[1] + fit_x[0] * x["cl_pos_z_cm"].values - x["cl_pos_x_cm"].values
    chi_x = np.sum(((fit_x[1] + fit_x[0] * x["cl_pos_z_cm"].values - x["cl_pos_x_cm"].values)**2)
             / fit_x[1] + fit_x[0] * x["cl_pos_z_cm"].values)

    fit_y = np.polyfit(x[(x.planar != put) & (x.cl_pos_y_cm.notna())]["cl_pos_z_cm"].dropna(), x[x.planar != put]["cl_pos_y_cm"].dropna(), 1)
    pos_y = fit_y[1] + fit_y[0] * x["cl_pos_z_cm"].values
    res_y = fit_y[1] + fit_y[0] * x["cl_pos_z_cm"].values - x["cl_pos_y_cm"].values
    chi_y = np.sum(((fit_y[1] + fit_y[0] * x["cl_pos_z_cm"].values - x["cl_pos_y_cm"].values)**2)
             / fit_y[1] + fit_y[0] * x["cl_pos_y_cm"].values)

    if tracking_fit:
        pos_x=np.insert(pos_x,put, np.nan)
        res_x=np.insert(res_x,put, np.nan)
        pos_y=np.insert(pos_y,put, np.nan)
        res_y=np.insert(res_y,put, np.nan)

    run = x["run"].values[0]
    subrun = x["subrun"].values[0]
    #     fig=px.scatter(x=x["cl_pos_z_cm"],y = x["cl_pos_x_cm"])
    #     fit_x=np.poly1d(fit)
    #     fig.add_trace( px.line(x=range(0,40),y=fit_x(range(0,40)) ).data[0])
    #     fig.add_trace( px.scatter(x=[0,10,20,30],y=pos ).data[0])
    #     fig.update_yaxes(range=[0, 9])
    #     fig.show()
    #     print (type(fit))
    #     print (type(pos))
    return pd.DataFrame(data=[[run, subrun, fit_x, pos_x, res_x, chi_x, fit_y, pos_y, res_y, chi_y]], columns=["run", "subrun", "fit_x", "pos_x", "res_x","chi_x", "fit_y", "pos_y", "res_y","chi_y"])


# def filter_tracks(tracks_pd, cut=0.2, res_max=0.7):
#     ## Filter the tracks before the correction calculation
#     tracks_pd_c = tracks_pd[
#         (tracks_pd["pos_x"].apply(lambda x: np.all(x < 8.32 - cut) & np.all(x > 0 + cut))) &
#         (tracks_pd["pos_y"].apply(lambda x: np.all(x < 8.32 - cut) & np.all(x > 0 + cut))) &
#         (tracks_pd["res_x"].apply(lambda x: np.all(abs(x) < res_max))) &
#         (tracks_pd["res_y"].apply(lambda x: np.all(abs(x) < res_max)))
#         ]
#     #     print (f"Dropped {len(tracks_pd)-len(tracks_pd_c)} tracks")
#     return tracks_pd_c


def calc_correction(trk_pd, planar=0):
    track_pd = trk_pd.copy()
    ## Calc the correction for a specific planar
    # Cast planar to int
    planar = int(planar)
    fit_dict = {}
    # Select data only for one planar
    track_pd["pos_x"] = track_pd["pos_x"].apply(lambda x: x[planar])
    track_pd["pos_y"] = track_pd["pos_y"].apply(lambda x: x[planar])
    track_pd["res_x"] = track_pd["res_x"].apply(lambda x: x[planar])
    track_pd["res_y"] = track_pd["res_y"].apply(lambda x: x[planar])
    ## Arrotondo al mm per fittare
    fit = np.polyfit(track_pd["pos_x"], track_pd["res_y"], 1)
    fit_x = (fit)
    fit_dict[f"{planar}_x"] = fit_x

    #     tracks_x=track_pd.groupby(f"pos_x")[f"res_y"].mean()
    #     fig=px.scatter(tracks_x, x=tracks_x.index,y = f"res_y")
    #     fig.add_trace( px.line(x=range(0,9),y=fit_x(range(0,9)) ).data[0])
    #     fig.show()

    #     track_pd[f"pos_y"]=((track_pd[f"pos_y"]*100).round())/100
    #     tracks_y=track_pd.groupby(f"pos_y")[f"res_x"].mean()
    #     tracks_w=(track_pd.groupby(f"pos_y")[f"res_x"].count())**(1/2)/track_pd.groupby(f"pos_x")[f"res_y"].std()
    #     try:
    fit = np.polyfit(track_pd[f"pos_y"], track_pd[f"res_x"], 1)
    #     except:
    #         print ("Exception!")
    #         print (tracks_w)
    fit_y = (fit)
    #         fig=px.scatter(tracks_y, x=tracks_y.index,y = f"res_planar_{planar}_x")
    #         fig.add_trace( px.line(x=range(0,9),y=fit_y(range(0,9)) ).data[0])
    #         fig.show()
    fit_dict[f"{planar}_y"] = fit_y

    return fit_dict

def apply_correction(cl_pd, correction):
    """
    Apply a correction on a 2D.pd
    :param cl_pd:
    :param planar:
    :param correction:
    :return:
    """
    sub_data = cl_pd.groupby(["run", "subrun"])
    sub_list = []
    return_list = []
    for key in sub_data.groups:
        sub_list.append(sub_data.get_group(key))
    if len(sub_list) > 0:
        with Pool(processes=20) as pool:
            with tqdm(total=len(sub_list), desc="Applying correction", leave=False) as pbar:
                for i, x in enumerate(pool.imap(apply_correction_fucn( correction), sub_list)):
                    return_list.append(x)
                    pbar.update()
        cl_pd = pd.concat(return_list)
    return cl_pd



class apply_correction_fucn(object):
    """
    Usign class function in order to specify arguments.
    """
    def __init__(self, correction):
        self.corrections = correction

    def __call__(self, cl_pd):
        cl_pd = cl_pd.apply(lambda x: apply_correction_process(x, self.corrections), axis=1)
        return cl_pd

def apply_correction_process(row, corrections):
    """
    Fucntion used to apply 2D correction to the dataset
    :param row:
    :param planar:
    :param correction:
    :return:
    """
    for n,correction in enumerate(corrections):
        angle = (correction[f"{int(row.planar)}_x"][0] - correction[f"{int(row.planar)}_y"][0]) / 2
        row.cl_pos_y_cm = row.cl_pos_y_cm + angle * (row.cl_pos_x_cm) + correction[f"{int(row.planar)}_x"][1]
        row.cl_pos_x_cm = row.cl_pos_x_cm - angle * (row.cl_pos_y_cm) + correction[f"{int(row.planar)}_y"][1]
    return row


def apply_correction_eff(row, epos_x, epos_y, corrections):
    """
    Fucntion used to apply 2D correction to the dataset
    :param row:
    :param planar:
    :param correction:
    :return:
    """
    for correction in corrections:
            angle = (correction[f"{int(row.planar)}_x"][0] - correction[f"{int(row.planar)}_y"][0]) / 2
            row.cl_pos_y_cm = row.cl_pos_y_cm + angle * (epos_x) + correction[f"{int(row.planar)}_x"][1]
            row.cl_pos_x_cm = row.cl_pos_x_cm - angle * (epos_y) + correction[f"{int(row.planar)}_y"][1]
    return row





# def double_gaus_fit(tracks_pd, view="x", put=-1):
#     popt_list = []
#     pcov_list = []
#     res_list = []
#     R_list = []
#
#     for pl in range(0, 4):
#         if pl==put:
#             popt_list.append(0)
#             pcov_list.append(0)
#             res_list.append(0)
#             R_list.append(1)
#         else:
#             data = tracks_pd[f"res_{view}"].apply(lambda x: x[pl])
#             sigma_0 = np.std(data)
#             if sigma_0<0.2:
#                 sigma_0=0.2
#             data = data[abs(data) < sigma_0]
#             sigma_0 = np.std(data)
#             y, x = np.histogram(data, bins=1000)
#             if y.max<200:
#                 y, x = np.histogram(data, bins=50)
#             mean_1 = x[np.argmax(y)]
#             mean_0 = x[np.argmax(y)]
#             a_0 = np.max(y)
#             a_1 = np.max(y) / 5
#             sigma_1 = sigma_0 * 3
#             x = (x[1:] + x[:-1]) / 2
#             upper_bound=[np.inf, 0.1, 1, np.inf,0.1,2,0]
#             lower_bound=[0,-0.1,0,0,-0.1,0,np.mean(y)]
#             popt, pcov = curve_fit(doublegaus, x, y, p0=[a_0, mean_0, sigma_0, a_1, mean_1, sigma_1, 0], bounds=(lower_bound, upper_bound))
#             popt_list.append(popt)
#             pcov_list.append(pcov)
#             yexp = doublegaus(x, *popt)
#             ss_res = np.sum((y - yexp) ** 2)
#             ss_tot = np.sum((y - np.mean(y)) ** 2)
#             res_list.append(y - yexp)
#             r2 = 1 - (ss_res / ss_tot)  # ynorm= 1000*y/np.sum(y)
#             R_list.append(r2)
#     #         yexp=doublegaus(x, *popt)
#     #         y_exp_norm =1000*yexp/np.sum(yexp)
#     #         print (np.sum(ynorm))
#     #         print (np.sum(y_exp_norm))
#     #         print (chisquare(ynorm,y_exp_norm, 6 ))
#     return popt_list, pcov_list, res_list, R_list

def load_nearest_correction(path,run_number, logger=None):
    path_list = glob.glob(os.path.join(path,"*"))
    corr_list = [int(path.split("/")[-1]) for path in path_list]
    corr_list = [corr for corr in corr_list if corr <= run_number]
    align_number = min(corr_list, key=lambda x:abs(x-run_number))
    if logger:
        logger.write_log(f"Nearest alignment correction: run {align_number}, Created: {time.ctime(os.path.getctime(os.path.join(path, str(align_number))))}")
    corr=load_correction(path, align_number)
    return corr


def load_correction(path, run_number):
    print(f"Nearest alignment correction: run {run_number}")
    #     print("Created: %s" % time.ctime(os.path.getctime(path)))
    with open(os.path.join(path, str(run_number)), 'rb') as corr_file:
        corr = pickle.load(corr_file)
    return corr







class calc_eff_func_class(object):
    """
    Using class function to initialize the efficiency calculator
    """

    def __init__(self, sigmas, residuals_dict, put, corrections, hit_efficiency=False):
        self.nsigma = sigmas
        self.res_dict = residuals_dict
        self.put = put
        self.corrections = corrections
        self.hit_efficiency = hit_efficiency
    def __call__(self, cl_and_tr):
        cl_pd_1d = cl_and_tr[0]
        tracks_pd = cl_and_tr[1]
        matching_cluster_pd, events_pd, eff_x, eff_y, tot_ev = calc_eff_process(tracks_pd, cl_pd_1d, self.res_dict, self.nsigma, self.put, corrections=self.corrections, hit_efficiency=self.hit_efficiency)
        return matching_cluster_pd, events_pd, eff_x, eff_y, tot_ev


def calc_eff_process(tracks_pd, cl_pd_1D, res_dict, nsimga_eff, put, corrections, hit_efficiency):
    """
    Process to calcualate the efficiency, returns a return blob
    """
    ##Inizializzo eff_pd
    count_l = []
    subrun_l = []
    epos_x_l = []
    epos_y_l = []
    eff_x_l = []
    eff_y_l = []
    run_l = []
    planar_l = []
    # Initialize the storage variables
    tot_ev = 0
    eff_x = 0
    eff_y = 0
    match_clusters = []
    # Extract from the dictionary the residuals parameters
    put_mean_x = res_dict["put_mean_x"]
    put_sigma_x = res_dict["put_sigma_x"]
    put_mean_y = res_dict["put_mean_y"]
    put_sigma_y = res_dict["put_sigma_y"]
    tracks_pd_c_event = tracks_pd.groupby(["count"])

    cl_pd_1D_event = cl_pd_1D.groupby(["count"])
    for event in tracks_pd_c_event.groups:
        #         if event in (cl_pd_1D_event.groups):
        eff_y_lb = False
        eff_x_lb = False
        tot_ev += 1
        # Selezione traccia e cluster di qeusto evento
        this_track = tracks_pd_c_event.get_group(event)
        clusters = cl_pd_1D_event.get_group(event)
        # Seleziono cluster della PUT (meglio fare qui per evitare di non trovare l'evento)
        clusters = clusters[clusters.planar == put]
        # Calcolo la posizione prevista sulla planare dalla traccia
        p_pos_x = this_track.fit_x.values[0][1] + this_track.fit_x.values[0][0] * (10 * put)
        p_pos_y = this_track.fit_y.values[0][1] + this_track.fit_y.values[0][0] * (10 * put)
        # Uso la posizione prevista per calcolare l'allineamento dei cluster 1D
        clusters = clusters.apply(lambda x: apply_correction_eff(x, p_pos_x, p_pos_y, corrections), axis=1)
        # x
        match_x = clusters[
            (clusters.cl_pos_x_cm - p_pos_x < put_mean_x + nsimga_eff * put_sigma_x) &
            (clusters.cl_pos_x_cm - p_pos_x > put_mean_x - nsimga_eff * put_sigma_x)]
        if match_x.shape[0] > 0:
            eff_x += 1
            eff_x_lb = True
            match_clusters.append(match_x)
        # y
        match_y = clusters[
            (clusters.cl_pos_y_cm - p_pos_y < put_mean_y + nsimga_eff * put_sigma_y) &
            (clusters.cl_pos_y_cm - p_pos_y > put_mean_y - nsimga_eff * put_sigma_y)]
        if match_y.shape[0] > 0:
            eff_y += 1
            eff_y_lb = True
            match_clusters.append(match_y)
        run_l.append(this_track.run.values[0])
        eff_x_l.append(eff_x_lb)
        eff_y_l.append(eff_y_lb)
        epos_x_l.append(p_pos_x)
        epos_y_l.append(p_pos_y)
        subrun_l.append(this_track.subrun.max())
        count_l.append(this_track["count"].max())
        planar_l.append(put)
    eff_pd = pd.DataFrame({
        "eff_x" : eff_x_l,
        "eff_y" : eff_y_l,
        "pos_x" : epos_x_l,
        "pos_y" : epos_y_l,
        "subrun": subrun_l,
        "count" : count_l,
        "run"   : run_l,
        "PUT": planar_l
    })
    return match_clusters, eff_pd, eff_x, eff_y, tot_ev

class log_writer():
    def __init__(self, path, run, log_name="logfile"):
        self.path = path
        self.run = run
        self.log_name=log_name
        with open(os.path.join(path, self.log_name), "w+") as logfile:
            pass

    def write_log(self, text):
        with open(os.path.join(str(self.path), self.log_name), "a") as logfile:
            logfile.write(text+"\n")

def calculte_eff(run, data_folder, put, cpu_to_use, nsigma_put=5, nsigma_trackers=1, chi_sq_trackers=0, multi_tracks_suppresion=False, hit_efficiency=False):
    runs = run
    #Create directories to store the outputs
    if not os.path.isdir(os.path.join(data_folder,"perf_out")):
        os.mkdir(os.path.join(data_folder,"perf_out"))
    if not os.path.isdir(os.path.join(data_folder,"perf_out",str(run))):
        os.mkdir(os.path.join(data_folder,"perf_out",str(run)))
    path_out_eff=os.path.join(data_folder,"perf_out",str(run))
    if not os.path.isdir(os.path.join(path_out_eff,"res_fit")):
        os.mkdir(os.path.join(path_out_eff,"res_fit"))
    logger=log_writer(path_out_eff, run)
    # Caricamento dei cluster 2D per i tracciatori
    cl_pd_2D_ori = load_cluster_2D_align(runs, data_folder)
    #Carica la correzione più vicina al run (minore del run)
    correction = load_nearest_correction(os.path.join(data_folder,"alignment"), runs, logger)
    # Applica la correzione dell'allineamento al cluster 2D
    cl_pd_2D = apply_correction(cl_pd_2D_ori, correction)
    if put==-1:
        put_list=[0,1,2,3]
    else:
        put_list=[put]
    for put in tqdm( put_list, desc="Planar"):
        print (f" Measuring performances on planar {put}")
        logger.write_log(f"-------\n Measuring performances on planar {put}")
        trackers_list = [0,1,2,3]
        trackers_list.remove(put)
        # Seleziona gli eventi con 4 cluster
        if multi_tracks_suppresion:
            cl_pd_2D = cl_pd_2D.groupby(["subrun", "count", "planar"]).filter(lambda x: x.shape[0]==1)  # Filtering away events with more than 1 cluster per view
        cl_pd_2D_res = cl_pd_2D.groupby(["subrun", "count"]).filter(lambda x: set(x["planar"]) == {0, 1, 2, 3})

        # Seleziona gli eventi che hanno i 3 tracciatori
        cl_pd_2D_tracking = cl_pd_2D.groupby(["subrun", "count"]).filter(lambda x: all([i in set(x["planar"]) for i in trackers_list]))
        # Fit them to extract the put sigma and mean
        tracks_pd = fit_tracks_manager(cl_pd_2D_tracking, put, True)
        ##Seleziona le tracce che rispettano l'intervallo di residui
        ##Seleziona le tracce che rispettano l'intervallo di residui
        nsigma_trck = nsigma_trackers
        tracks_pd_c = tracks_pd
        # tracks_pd_c.drop_duplicates(inplace=True)

        logger.write_log(f"{tracks_pd_c.shape[0]} tracks with all trackres before cutting")

        for view in ("x", "y"):
            popt_list, pcov_list, res_list, R_list,chi_list, deg_list = r_fit.double_gaus_fit_root(tracks_pd, view, put)


            for pl in trackers_list:
                mean_res = ((popt_list[pl][1] * popt_list[pl][0] * popt_list[pl][2]) + (popt_list[pl][4] * popt_list[pl][3] * popt_list[pl][5])) / (popt_list[pl][0] * popt_list[pl][2] + popt_list[pl][3] * popt_list[pl][5])
                res_sigma = ((popt_list[pl][2] * popt_list[pl][0] * popt_list[pl][2]) + (popt_list[pl][5] * popt_list[pl][3] * popt_list[pl][5])) / (popt_list[pl][0] * popt_list[pl][2] + popt_list[pl][3] * popt_list[pl][5])
                r_fit.plot_residuals(tracks_pd, view, popt_list, R_list, path_out_eff, put, mean_res, res_sigma, nsigma_trck, pl, chi_list, deg_list)
                # print(f"mean {mean_res},sigma {nsigma_trck*res_sigma} ")
                # print (tracks_pd_c[f"res_{view}"].apply(lambda x: x[pl]))
                logger.write_log("Trackers fits")
                logger.write_log(f"pl {pl}, view {view}, mean {mean_res}, res_sigma {res_sigma}")
                tracks_pd_c = tracks_pd_c[
                    (tracks_pd_c[f"res_{view}"].apply(lambda x: x[pl]) > (mean_res - nsigma_trck*res_sigma)) &
                    (tracks_pd_c[f"res_{view}"].apply(lambda x: x[pl]) < (mean_res + nsigma_trck*res_sigma))
                    ]
            if any([R < 0.85 for R in R_list]):
                logger.write_log(
                    f"One R2 in  trackers fit is less than 0.9,  verify the fits on view {view}, put {put}")
                raise Warning(f"One R2 in  trackers fit is less than 0.9,  verify the fits on view {view}, put {put}")
        good_events = tracks_pd_c["count"].unique()
        # Fitta le tracce
        cl_pd_2D_res = cl_pd_2D_res[cl_pd_2D_res["count"].isin(good_events)]# Solo degli eventi con tracciatori buoni
        tracks_pd_res = fit_tracks_manager(cl_pd_2D_res, put)

        # Estraggo mean e sigma sulla planare sotto test, serve per stabilire l'efficienza
        view = "x"
        popt_list, pcov_list, res_list, R_list,chi_list, deg_list = r_fit.double_gaus_fit_root(tracks_pd_res, view)
        # print (len(popt_list), len(pcov_list), len(res_list), len(R_list),len(chi_list), len(deg_list))
        put_mean_x = ((popt_list[0][1] * popt_list[0][0] * popt_list[0][2]) + (popt_list[0][4] * popt_list[0][3] * popt_list[0][5])) / (popt_list[0][0] * popt_list[0][2] + popt_list[0][3] * popt_list[0][5])
        put_sigma_x = ((popt_list[0][2] * popt_list[0][0] * popt_list[0][2]) + (popt_list[0][5] * popt_list[0][3] * popt_list[0][5])) / (popt_list[0][0] * popt_list[0][2] + popt_list[0][3] * popt_list[0][5])
        popt_list_put_x=popt_list
        r_fit.plot_residuals(tracks_pd_res, view, popt_list, R_list*4, path_out_eff, put, put_mean_x, put_sigma_x, nsigma_put, put, chi_list, deg_list)

        if any([R < 0.85 for R in R_list]):
            logger.write_log(f"One R2 in PUT fit is less than 0.85,  verify the fits on view {view}, put {put}")
            raise Warning(f"One R2 in PUT fit is less than 0.85,  verify the fits on view {view}, put {put}")


        view = "y"
        popt_list, pcov_list, res_list, R_list,chi_list, deg_list = r_fit.double_gaus_fit_root(tracks_pd_res, view)
        put_mean_y = ((popt_list[0][1] * popt_list[0][0] * popt_list[0][2]) + (popt_list[0][4] * popt_list[0][3] * popt_list[0][5])) / (popt_list[0][0] * popt_list[0][2] + popt_list[0][3] * popt_list[0][5])
        put_sigma_y = ((popt_list[0][2] * popt_list[0][0] * popt_list[0][2]) + (popt_list[0][5] * popt_list[0][3] * popt_list[0][5])) / (popt_list[0][0] * popt_list[0][2] + popt_list[0][3] * popt_list[0][5])
        popt_list_put_y=popt_list
        r_fit.plot_residuals(tracks_pd_res, view, popt_list, R_list*4, path_out_eff, put, put_mean_y, put_sigma_y, nsigma_put, put, chi_list, deg_list)

        if any([R < 0.85 for R in R_list]):
            logger.write_log(f"One R2 in PUT fit is less than 0.85,  verify the fits on view {view}, put {put}")
            raise Warning(f"One R2 in PUT fit is less than 0.85, verify the fits on view {view}, put {put}")

        logger.write_log(f"Pl{put}, sigma_x{put_sigma_x}, sigma_y{put_sigma_y}")

        logger.write_log(f"Measuring efficiency on {tracks_pd_c.shape[0]} tracks")

        # Carico i cluster 1D per misurare l'efficienza
        cl_pd_1D = get_run_data([runs], '1D', data_folder)
        # cl_pd_1D=cl_pd_1D[cl_pd_1D.planar==put]
        ## Carico la posizione in cm non allineata
        cl_pd_1D["cl_pos_x_cm"] = cl_pd_1D.cl_pos_x * 0.0650
        cl_pd_1D["cl_pos_y_cm"] = cl_pd_1D.cl_pos_y * 0.0650
        cl_pd_1D["cl_pos_z_cm"] = cl_pd_1D.planar * 10
        tracks_pd_c_sub = tracks_pd_c.groupby(["subrun"])
        cl_pd_1D_sub = cl_pd_1D.groupby(["subrun"])
        residuals_dict = {
            "put_mean_x" : put_mean_x,
            "put_sigma_x": put_sigma_x,
            "put_mean_y" : put_mean_y,
            "put_sigma_y": put_sigma_y
        }
        sub_list = []
        return_list = []

        par_for_int = popt_list_put_x[put]
        par_for_int[6] = 0
        integral_x = scipy.integrate.quad(r_fit.doublegaus, -0.5, 0.5,args=(tuple(par_for_int)))[0]
        this_x_int = scipy.integrate.quad(r_fit.doublegaus, put_mean_x-put_sigma_x*nsigma_put,put_mean_x+put_sigma_x*nsigma_put,args=(tuple(par_for_int)))[0]

        par_for_int = popt_list_put_y[put]
        par_for_int[6] = 0
        integral_y = scipy.integrate.quad(r_fit.doublegaus, -0.5, 0.5,args=(tuple(par_for_int)))[0]
        this_y_int = scipy.integrate.quad(r_fit.doublegaus, put_mean_y-put_sigma_y*nsigma_put,put_mean_y+put_sigma_y*nsigma_put,args=(tuple(par_for_int)))[0]
        logger.write_log(f"Residual x tolerance on DUT:{put_mean_x:.4f}+/-{put_sigma_x*nsigma_put:.3f} {this_x_int/integral_x*100}% of total integral"
                         f"\nResidual y tolerance on DUT: {put_mean_y:.4f}+/-{put_sigma_y*nsigma_put:.3f} {this_y_int/integral_y*100}% of total integral\n")
        if hit_efficiency:
            config = configparser.ConfigParser()
            config.read(os.path.join(sys.path[0], "config.ini"))
            signal_window_lower_limit_conf = int(config["GLOBAL"].get("signal_window_lower_limit"))
            signal_window_upper_limit_conf = int(config["GLOBAL"].get("signal_window_upper_limit"))
            cl_pd_1D = get_run_data([runs], 'h', data_folder)
            cl_pd_1D=cl_pd_1D[(cl_pd_1D["l1ts_min_tcoarse"] > signal_window_lower_limit_conf) & (cl_pd_1D["l1ts_min_tcoarse"]<signal_window_upper_limit_conf)]
            cl_pd_1D_sub = cl_pd_1D.groupby(["subRunNo"])
            cl_pd_1D["cl_pos_x_cm"] = cl_pd_1D[cl_pd_1D.strip_x > -1].strip_x * 0.0650
            cl_pd_1D["cl_pos_y_cm"] = cl_pd_1D[cl_pd_1D.strip_y > -1].strip_y * 0.0650
        for key in tracks_pd_c_sub.groups:
            sub_list.append((cl_pd_1D_sub.get_group(key), tracks_pd_c_sub.get_group(key)))
        if len(sub_list) > 0:
            with Pool(processes=cpu_to_use) as pool:
                with tqdm(total=len(sub_list), desc="Calculating event efficiency", leave=False) as pbar:
                    for i, x in enumerate(pool.imap(calc_eff_func_class(sigmas=nsigma_put, residuals_dict=residuals_dict, put=put, corrections=correction), sub_list)):
                        return_list.append(x)
                        pbar.update()

        eff_x = np.sum([x [2] for x in return_list])
        eff_y = np.sum([x [3] for x in return_list])
        tot_ev = np.sum([x [4] for x in return_list])
        print(f"-Eff dut {put}:\n X:{eff_x/tot_ev} Y:{eff_y/tot_ev}\n")
        logger.write_log(f"-Eff dut {put}:\n X:{eff_x/tot_ev} Y:{eff_y/tot_ev}\n")


        cl_list=[]

        with Pool(processes=cpu_to_use) as pool:
            with tqdm(total=len(sub_list), desc="Merging return cluster data", leave=False) as pbar:
                for i, x in enumerate(pool.imap(concat_subrun_cluster, return_list)):
                    cl_list.append(x)
                    pbar.update()

        # for x in tqdm(return_list, desc="Merging return cluster data"):
        #     if len(x[0])>0:
        #         cl_list.append(pd.concat(x[0]))

        if len(cl_list)>0:
            matching_clusters = pd.concat(cl_list)
            matching_clusters.to_pickle(os.path.join(path_out_eff, f"match_cl_{put}.gzip"), compression="gzip")

        eff_pd = pd.concat([x[1] for x in return_list])
        eff_pd.to_pickle(os.path.join(path_out_eff, f"eff_pd_{put}.gzip"), compression="gzip")
        eff_pd_c = eff_pd
        eff_pd["pos_x_pl"], eff_pd["pos_y_pl"] = zip(*eff_pd.apply(lambda x: de_correct_process(x, correction), axis=1))
        k = eff_pd_c[(eff_pd_c.eff_x) & (eff_pd_c.pos_x_pl > 3.2) & (eff_pd_c.pos_x_pl < 7.8) & (eff_pd_c.pos_y_pl > 3.2) & (eff_pd_c.pos_y_pl < 7.8)].count().eff_x
        n = eff_pd_c[(eff_pd_c.pos_x_pl > 3.2) & (eff_pd_c.pos_x_pl < 7.8) & (eff_pd_c.pos_y_pl > 3.2) & (eff_pd_c.pos_y_pl < 7.8)].count().eff_x
        eff_x_good = k / n
        eff_x_good_error = (((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) ** 2) / ((n + 2) ** 2)) ** (1 / 2)
        print("Efficiency in range [3.2,7.8]\n")
        logger.write_log(f"Efficiency in range [3.2,7.8]\n")
        print(f"X: {eff_x_good:.4f} +/- {eff_x_good_error:.4f}\n")
        logger.write_log(f"X: {eff_x_good:.4f} +/- {eff_x_good_error:.4f}\n")

        k = eff_pd_c[(eff_pd_c.eff_y) & (eff_pd_c.pos_y_pl > 3.2) & (eff_pd_c.pos_y_pl < 7.8) & (eff_pd_c.pos_x_pl > 3.2) & (eff_pd_c.pos_x_pl < 7.8)].count().eff_y
        n = eff_pd_c[(eff_pd_c.pos_y_pl > 3.2) & (eff_pd_c.pos_y_pl < 7.8) & (eff_pd_c.pos_x_pl > 3.2) & (eff_pd_c.pos_x_pl < 7.8)].count().eff_y
        eff_y_good = k / n
        eff_y_good_error = (((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) ** 2) / ((n + 2) ** 2)) ** (1 / 2)
        logger.write_log(f"Y: {eff_y_good:.4f} +/- {eff_y_good_error:.4f}\n")
        print(f"Y: {eff_y_good:.4f} +/- {eff_y_good_error:.4f}\n")

        tracks_pd.to_pickle(os.path.join(path_out_eff, f"tracks_pd_{put}.gzip"), compression="gzip")
        del eff_pd
        del tracks_pd
        del cl_pd_1D

def concat_subrun_cluster(cl_list):
    if len(cl_list[0])>0:
        return (pd.concat(cl_list[0]))

def de_correct_process(row, corr):
    planar = row.PUT
    pos_x = row.pos_x
    pos_y = row.pos_y
    rev_corr = corr[::-1]
    for correction in corr:
        angle = (correction[f"{int(planar)}_x"][0] - correction[f"{int(planar)}_y"][0]) / 2
        pos_x = pos_x - correction[f"{int(planar)}_y"][1] + angle * (pos_y)
        pos_y = pos_y - correction[f"{int(planar)}_x"][1] - angle * (pos_x)
    return pos_x, pos_y