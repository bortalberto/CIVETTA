import pickle
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from multiprocessing import Pool

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

class alignment_class():
    """
    Class to hold the alignment setup
    """
    def __init__(self, cpu, rounds):
        self.cput_to_use=cpu
        self.rounds=rounds
        self.corrections=[]

    def load_cluster_2D_align(self, runs, data_folder, downsampling):
        """
        Loads the 2D clusters and keeps only the ones with clusters on 4 planars
        :param runs:
        :param data_folder:
        :return:
        """
        #Load cluster data
        cl_pd_2D=get_run_data([runs],'2D', data_folder)
        #Calculate standard position
        cl_pd_2D["cl_pos_x_cm"] = cl_pd_2D.cl_pos_x * 0.0650
        cl_pd_2D["cl_pos_y_cm"] = cl_pd_2D.cl_pos_y * 0.0650
        cl_pd_2D["cl_pos_z_cm"] = cl_pd_2D.planar * 10
        #Drop old position to save memory
        cl_pd_2D=cl_pd_2D.drop(columns=["cl_pos_x","cl_pos_y"])
        #Drop charge and size position, not needed for alinment
        cl_pd_2D=cl_pd_2D.drop(columns=["cl_charge","cl_charge_x","cl_charge_y","cl_size_x","cl_size_y","cl_size_tot"])
        #Let's keep only events with 4 planars
        cl_pd_2D=cl_pd_2D.groupby(["subrun","count"]).filter(lambda x: set(x["planar"])=={0,1,2,3})
        if downsampling!=1:
            cl_pd_2D=cl_pd_2D[cl_pd_2D["count"] % downsampling == 0]
        return cl_pd_2D


    def fit_tracks_manager(self, cl_pd, planar="None"):
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
                    for i, x in enumerate(pool.imap_unordered(fit_tracks_process_pd(planar), sub_list)):
                        return_list.append(x)
                        pbar.update()
            track_pd = pd.concat(return_list)
        else:
            print ( " No suburns to calibrate")
            exit()
            track_pd=pd.DataFrame()
        track_pd = track_pd.reset_index()
        # track_pd = track_pd.drop(columns="level_1")

        return track_pd






    def filter_tracks(self, tracks_pd, cut=0.2, res_max=0.7):
        """
        Filter the tracks to improve the alignment quality
        :param tracks_pd:
        :param cut:
        :param res_max:
        :return:
        """
        ## Filter the tracks before the correction calculation
        tracks_pd_c = tracks_pd[
            (tracks_pd["pos_x"].apply(lambda x: np.all(x < 8.32 - cut) & np.all(x > 0 + cut))) &
            (tracks_pd["pos_y"].apply(lambda x: np.all(x < 8.32 - cut) & np.all(x > 0 + cut))) &
            (tracks_pd["res_x"].apply(lambda x: np.all(abs(x) < res_max))) &
            (tracks_pd["res_y"].apply(lambda x: np.all(abs(x) < res_max)))
            ]
        #     print (f"Dropped {len(tracks_pd)-len(tracks_pd_c)} tracks")
        return tracks_pd_c


    def calc_correction(self, trk_pd, planar=0):
        """
        Calc the correction for a specific planar, taking a tracks_pd and a PUT. Performs a linear fit
        :param trk_pd:
        :param planar:
        :return:
        """
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
        fit_x = fit
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
        fit_y = fit
        #         fig=px.scatter(tracks_y, x=tracks_y.index,y = f"res_planar_{planar}_x")
        #         fig.add_trace( px.line(x=range(0,9),y=fit_y(range(0,9)) ).data[0])
        #         fig.show()
        fit_dict[f"{planar}_y"] = fit_y

        return fit_dict


    def apply_correction(self, cl_pd, planar, correction):
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
            with Pool(processes=self.cput_to_use) as pool:
                with tqdm(total=len(sub_list), desc="Applying correction", leave=False) as pbar:
                    for i, x in enumerate(pool.imap(apply_correction_fucn(planar, correction), sub_list)):
                        return_list.append(x)
                        pbar.update()
            cl_pd = pd.concat(return_list)
        return cl_pd




    def save_corrections(self, data_folder, run):
        """
        Save the calculated corrections
        :return:
        """
        # Do not compress the correction in one step
        # correction = {}
        # for view in ("x", "y"):
        #     for planar in range(0, 4):
        #         correction[f"{planar}_{view}"] = np.poly1d([0, 0])
        # for elem in self.corrections:
        #     for key in elem:
        #         correction[key] += elem[key]


        with open(os.path.join(data_folder,"alignment", str(run)), 'wb+') as ali_file:
            pickle.dump(self.corrections, ali_file)





class apply_correction_fucn(object):
    """
    Usign class function in order to specify arguments.
    """
    def __init__(self, planar, correction):
        self.target_planar = planar
        self.correction = correction

    def __call__(self, cl_pd):
        cl_pd = cl_pd.apply(lambda x: apply_correction_process(x, self.target_planar, self.correction), axis=1)
        return cl_pd

def apply_correction_process(row, planar, correction):
    """
    Fucntion used to apply 2D correction to the dataset
    :param row:
    :param planar:
    :param correction:
    :return:
    """
    if int(row.planar) == int(planar):
        angle = (correction[f"{int(row.planar)}_x"][1] - correction[f"{int(row.planar)}_y"][1]) / 2
        row.cl_pos_y_cm = row.cl_pos_y_cm + angle * (row.cl_pos_x_cm) + correction[f"{int(row.planar)}_y"][0]
        row.cl_pos_x_cm = row.cl_pos_x_cm - angle * (row.cl_pos_y_cm) + correction[f"{int(row.planar)}_x"][0]

    return row

class fit_tracks_process_pd(object):
    def __init__(self, planar):
        self.put = planar

    def __call__(self, cl_pd):
        cl_pd = cl_pd.reset_index()
        if self.put == "None":
            tracks_pd = cl_pd.groupby(["count"])[["run", "subrun", "cl_pos_x_cm", "cl_pos_y_cm", "cl_pos_z_cm", "planar"]].apply(fit_tracks_process_row)
        else:
            tracks_pd = cl_pd.groupby(["count"])[["run", "subrun", "cl_pos_x_cm", "cl_pos_y_cm", "cl_pos_z_cm", "planar"]].apply(lambda x: fit_tracks_process_row(x, self.put))
        return tracks_pd

def fit_tracks_process_row( x, put="None"):
    x = x.sort_values("planar")
    fit_x = np.polyfit(x[ x.planar != put ]["cl_pos_z_cm"], x[ x.planar != put ]["cl_pos_x_cm"], 1)
    pos_x = fit_x[1] + fit_x[0] * x["cl_pos_z_cm"].values
    res_x = fit_x[1] + fit_x[0] * x["cl_pos_z_cm"].values - x["cl_pos_x_cm"].values
    x = x.sort_values("planar")
    fit_y = np.polyfit(x[ x.planar != put ]["cl_pos_z_cm"], x[ x.planar != put ]["cl_pos_y_cm"], 1)
    pos_y = fit_y[1] + fit_y[0] * x["cl_pos_z_cm"].values
    res_y = fit_y[1] + fit_y[0] * x["cl_pos_z_cm"].values - x["cl_pos_y_cm"].values
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
    return pd.DataFrame(data=[[run, subrun, fit_x, pos_x, res_x, fit_y, pos_y, res_y]], columns=["run", "subrun", "fit_x", "pos_x", "res_x", "fit_y", "pos_y", "res_y"])

def calibrate_alignment_run(run, rounds, cpu, data_folder, downsampling):
    alignment_istance = alignment_class(cpu=cpu, rounds=rounds)
    cl_pd_2D = alignment_istance.load_cluster_2D_align(run, data_folder, downsampling)
    # tracks_pd = alignment_istance.fit_tracks_manager(cl_pd_2D)
    # tracks_pd = alignment_istance.filter_tracks(tracks_pd)
    for it in tqdm(range (0,rounds), desc="Cycles"):
        correction={}
        for view in ("x","y"):
            for planar in range (0,4):
                 correction[f"{planar}_{view}"]=np.poly1d([0,0])
        for pl in tqdm([3,2,1], leave = False , desc="Planars"):
            tracks_pd = alignment_istance.fit_tracks_manager(cl_pd_2D,pl)
            # print(tracks_pd.shape)
            tracks_pd = alignment_istance.filter_tracks(tracks_pd, cut=-0.2)
            # print(tracks_pd.shape)
            correction.update(alignment_istance.calc_correction(tracks_pd, planar=pl))
            cl_pd_2D = alignment_istance.apply_correction(cl_pd_2D, pl, correction)
        alignment_istance.corrections.append(correction)
    alignment_istance.save_corrections(data_folder, run)
    # tracks_pd = alignment_istance.fit_tracks_manager(cl_pd_2D)
    # tracks_pd = alignment_istance.filter_tracks(tracks_pd)
