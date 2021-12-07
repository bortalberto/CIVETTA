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
    def __init__(self, data_folder,run,calib_folder,mapping_file,cpu_to_use=cpu_count(), Silent=False , purge=True, alignment=False, root=False, downsampling=1, cylinder=False, data_folder_root="Default" ):
        self.data_folder = data_folder
        if data_folder_root == "Default":
            self.data_folder_root=os.path.join(data_folder,"raw_root")
        else:
            self.data_folder_root=data_folder_root
        self.calib_folder = calib_folder
        self.mapping_file = mapping_file
        self.cpu_to_use = cpu_to_use
        self.run_number = run
        self.silent = Silent
        self.purge = purge
        self.alignment = alignment
        self.root = root
        self.cylinder = cylinder
        self.downsampling=downsampling

    ################# Decode part #################
    def decode_on_file(self,input_):
        self.decoder.decode_file(input_, self.root)
        filename = input_[0]
        # os.rename(filename.split(".")[0]+".root",self.data_folder+"/raw_root/{}/".format(self.run_number)+filename.split("/")[-1].split(".")[0]+".root")

    def dec_subrun(self, subrun_tgt):
        input_list=[]
        self.decoder = pl_lib.decoder(1, self.run_number, downsamplig=self.downsampling)
        for filename ,(subrun,gemroc) in glob2.iglob(self.data_folder+"/raw_dat/RUN_{}/SubRUN_*_GEMROC_*_TM.dat".format(self.run_number), with_matches=True):
            input_list.append((filename, int(subrun), int(gemroc)))
        for input in input_list:
            if f"SubRUN_{subrun_tgt}_" in str(input[0]):
                self.decode_on_file(input)
        if not self.silent:
            print (f"Subrun {subrun_tgt} decoded")

    def dec_run(self):
        """
        Decodes one run in parallel
        :return:
        """
        input_list=[]
        self.decoder = pl_lib.decoder(1, self.run_number, downsamplig=self.downsampling)
        if not self.silent:
            print ("Decoding")
        for filename ,(subrun,gemroc) in glob2.iglob(self.data_folder+"/raw_dat/RUN_{}/SubRUN_*_GEMROC_*_TM.dat".format(self.run_number), with_matches=True):
            input_list.append((filename, int(subrun), int(gemroc)))
        if len(input_list)>0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(input_list), disable=self.silent) as pbar:
                    for i, _ in enumerate(pool.imap_unordered(self.decode_on_file, input_list)):
                        pbar.update()
        else:
            print (f"Can't find any .dat file in {self.data_folder}/raw_dat/RUN_{self.run_number}")

    def dec_run_fill(self,subrun_tgt):
        """
        Decodes all the subruns wich are not already done up to the subrun target (checks in the hit_data pickle files)
        :param subrun_tgt:
        :return:
        """
        subrun_list=[]
        input_list=[]
        self.decoder = pl_lib.decoder(1, self.run_number, downsamplig=self.downsampling)

        path=self.data_folder+f"/raw_root/{self.run_number}/hit_data.pickle.gzip"
        if not self.silent:
            print (f"Decoding filling up to subrun {subrun_tgt}")
        if os.path.isfile(path):
            data_pd=pd.read_pickle(path,compression="gzip")
            done_subruns=data_pd.subRunNo.unique()
        else:
            done_subruns=[]
        for filename ,(subrun,gemroc) in glob2.iglob(self.data_folder+"/raw_dat/RUN_{}/SubRUN_*_GEMROC_*_TM.dat".format(self.run_number), with_matches=True):
            input_list.append((filename, int(subrun), int(gemroc)))
            subrun_list.append(int(subrun))

        subruns_to_do = (set(subrun_list) - set(done_subruns))
        subruns_to_do = [x for x in subruns_to_do if x <= subrun_tgt]

        if len(subruns_to_do) == 0:
            print ("Nothing to decode")
            return (0)
        input_list = [(f,s,g) for f,s,g in input_list if s in subruns_to_do]
        if len(input_list)>0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(input_list), disable=self.silent) as pbar:
                    for i, _ in enumerate(pool.imap_unordered(self.decode_on_file, input_list)):
                        pbar.update()
        else:
            print (f"Can't find any .dat file in {self.data_folder}/raw_dat/RUN_{self.run_number}")

    def clean_for_empty_subrun(self):
        #TODO: to complete
        input_list=[]
        for filename ,(subrun,gemroc) in glob2.iglob(self.data_folder+"/raw_dat/RUN_{}/SubRUN_*_GEMROC_*_TM.dat".format(self.run_number), with_matches=True):
            input_list.append((filename, int(subrun), int(gemroc)))

    def merge_dec (self):
        """
        Merge the decoded files GEMROC-wise and delete the metafiles
        :return:
        """
        subrun_list = []
        file_list = []
        pd_list=[]
        if self.root:
            pass
            # for filename ,(subrun,gemroc) in glob2.iglob(self.data_folder+"/raw_root/{}/SubRUN_*_GEMROC_*_TM.root".format(self.run_number), with_matches=True):
            #     subrun_list.append(subrun)
            #     file_list.append(filename)
            # if not self.silent:
            #     print ("Merging files")
            # for subrun in tqdm(set(subrun_list), disable=self.silent):
            #    os.system("hadd -f {0}/raw_root/{1}/Sub_RUN_dec_{2}.root {0}/raw_root/{1}/SubRUN_{2}_GEMROC_*_TM.root  >/dev/null 2>&1".format(self.data_folder,self.run_number,subrun))
        else:
            for filename ,(subrun,gemroc) in glob2.iglob(self.data_folder+"/raw_root/{}/SubRUN_*_GEMROC_*_TM.pickle.gzip".format(self.run_number), with_matches=True):
                subrun_list.append(subrun)
                file_list.append(filename)
            if not self.silent:
                print ("Merging files")

            for subrun in tqdm(set(subrun_list), disable=self.silent):
                pd_list = []
                for filename, (gemroc) in glob2.iglob(self.data_folder + "/raw_root/{}/SubRUN_{}_GEMROC_*_TM.pickle.gzip".format(self.run_number, subrun), with_matches=True):
                    pd_list.append(pd.read_pickle(filename, compression="gzip"))
                subrun_pd=pd.concat(pd_list,ignore_index=True)
                subrun_pd.to_pickle(self.data_folder + "/raw_root/{}/Sub_RUN_dec_{}.pickle.gzip".format(self.run_number, subrun), compression="gzip")


        if self.purge:
            if not self.silent:
                print ("Removing garbage files")
            for filen in file_list:
                os.remove(filen)

    ################# Ana part #################
    def get_dec_list(self):
        """
        Generates the dec list accordingly to the root option
        :return:
        """

        subrun_list = []
        file_list = []
        if self.root:
            for filename, subrun in glob2.iglob(self.data_folder + "/raw_root/{}/Sub_RUN_dec_*.root".format(self.run_number), with_matches=True):
                subrun_list.append(subrun[0])  # subrun is a tuple
                file_list.append(filename)

        else:
            for filename, subrun in glob2.iglob(self.data_folder + "/raw_root/{}/Sub_RUN_dec_*.pickle.gzip".format(self.run_number), with_matches=True):
                subrun_list.append(int(subrun[0]))  # subrun is a tuple
                file_list.append(filename)

        return (subrun_list,file_list)
    def calib_run(self):
        analizer = pl_lib.calib(run_number=self.run_number, calib_folder=self.calib_folder, mapping_file=self.mapping_file, data_folder=self.data_folder, root_dec=self.root, cylinder=self.cylinder)
        analizer.load_mapping()
        subrun_list, file_list = self.get_dec_list()

        if len(subrun_list) > 0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(subrun_list), disable=self.silent) as pbar:
                    for i, _ in enumerate(pool.imap_unordered(analizer.calibrate_subrun, subrun_list)):
                        pbar.update()
        else:
            print(f"Can't find any decoded file in {self.data_folder}raw_root/{self.run_number}")
        analizer.create_hits_pd_and_single_root()

        if self.purge:
            if not self.silent:
                print("Removing garbage files (both dec and ana)")
            for filen in file_list:
                os.remove(filen)
            file_list = []
            for filename, subrun in glob2.iglob(self.data_folder + "/raw_root/{}/Sub_RUN_pl_ana*".format(self.run_number), with_matches=True):
                file_list.append(filename)
            for filen in file_list:
                os.remove(filen)


    def calib_run_fill(self,subrun_tgt):
        """
        Calibrate all the subruns wich are not already done up to the subrun target (checks in the hit_data pickle files)
        :param subrun_tgt:
        :return:
        """
        path = self.data_folder + f"/raw_root/{self.run_number}/hit_data.pickle.gzip"
        if not self.silent:
            print(f"Calibrating filling up to subrun {subrun_tgt}")
        if os.path.isfile(path):
            data_pd = pd.read_pickle(path, compression="gzip")
            done_subruns = data_pd.subRunNo.unique()
        else:
            done_subruns = []

        analizer = pl_lib.calib(run_number=self.run_number, calib_folder=self.calib_folder, mapping_file=self.mapping_file, data_folder=self.data_folder, root_dec=self.root,cylinder=self.cylinder)
        analizer.load_mapping()

        subrun_list, file_list = self.get_dec_list()


        subruns_to_do = (set(subrun_list) - set(done_subruns))
        subrun_list = [x for x in subruns_to_do if x <= subrun_tgt]
        if len(subrun_list) > 0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(subrun_list), disable=self.silent) as pbar:
                    for i, _ in enumerate(pool.imap_unordered(analizer.calibrate_subrun, subrun_list)):
                        pbar.update()
        else:
            print(f"Can't find any decoded file to calibrate in {self.data_folder}raw_root/{self.run_number}")

        analizer.append_hits_pd_and_single_root()

        if self.purge:
            if not self.silent:
                print("Removing garbage files (both dec and ana)")
            for filen in file_list:
                os.remove(filen)
            file_list = []
            for filename, subrun in glob2.iglob(self.data_folder + "/raw_root/{}/Sub_RUN_pl_ana*.root".format(self.run_number), with_matches=True):
                file_list.append(filename)
            for filen in file_list:
                os.remove(filen)


    def calib_subrun(self,subrun_tgt):
        analizer = pl_lib.calib(run_number=self.run_number, calib_folder=self.calib_folder, mapping_file=self.mapping_file, data_folder=self.data_folder, root_dec=self.root,cylinder=self.cylinder)
        analizer.load_mapping()

        subrun_list, file_list = self.get_dec_list()

        if subrun_tgt not in subrun_list:
            print(f"Can't find the decoded files in {self.data_folder}raw_root/{self.run_number}, try lo launch -d too")
        for sub in subrun_list:
            if sub == subrun_tgt:
                analizer.calibrate_subrun(sub)


    ################# Cls part #################
    def clusterize_run(self, time_limits):
        """
        Clusterize one run
        :param run_number:
        :param data_folder:
        :return:
        """
        pd_1d_return_list=[]
        pd_2d_return_list=[]
        if time_limits:
            clusterizer=pl_lib.clusterize(self.run_number, self.data_folder,time_limits[0], time_limits[1])
        else:
            clusterizer=pl_lib.clusterize.default_time_winw(self.run_number, self.data_folder)
        clusterizer.load_data_pd()
        sub_data = clusterizer.data_pd.groupby("subRunNo")
        sub_list = []
        for key in sub_data.groups:
            sub_list.append(sub_data.get_group(key))
        del clusterizer.data_pd
        if not self.silent:
            print ("Single view")
        if len(sub_list)>0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(sub_list), disable=self.silent) as pbar:
                    for i, x in enumerate(pool.imap_unordered(clusterizer.build_view_clusters, sub_list)):
                        pd_1d_return_list.append(x)
                        pbar.update()
            clusterizer.cluster_pd=pd.concat(pd_1d_return_list)
        else:
            print ("No subrun to clusterize, is the file hit_data.pickle.gzip in the working folder? Try to launch with -a")
            return (1)
        clusterizer.save_cluster_pd()
        if not self.cylinder:
            sub_data = clusterizer.cluster_pd.groupby("subrun")
            sub_list = []
            for key in sub_data.groups:
                sub_list.append(sub_data.get_group(key))
            if not self.silent:
                print ("Clusters 2-D")
            if len(sub_list) > 0:
                with Pool(processes=self.cpu_to_use) as pool:
                    with tqdm(total=len(sub_list), disable=self.silent) as pbar_2:
                        for i, x in enumerate(pool.imap_unordered(clusterizer.build_2D_clusters, sub_list)):
                            pd_2d_return_list.append(x)
                            pbar_2.update()
                clusterizer.cluster_pd_2D = pd.concat(pd_2d_return_list)
            clusterizer.save_cluster_pd_2D()

    def clusterize_run_fill(self, subrun_tgt,time_limits):
        """
        Clusterize one run
        :param run_number:
        :param data_folder:
        :return:
        """
        pd_1d_return_list=[]
        pd_2d_return_list=[]
        if time_limits:
            clusterizer=pl_lib.clusterize(self.run_number, self.data_folder,time_limits[0], time_limits[1])
        else:
            clusterizer=pl_lib.clusterize.default_time_winw(self.run_number, self.data_folder)
        clusterizer.load_data_pd()
        if not self.silent:
            print(f"Clusterizing filling up to subrun {subrun_tgt}")

        path = self.data_folder + f"/raw_root/{self.run_number}/cluster_pd_1D.pickle.gzip"
        if os.path.isfile(path):
            data_pd = pd.read_pickle(path, compression="gzip")
            done_subruns = data_pd.subrun.unique()
        else:
            done_subruns = []

        subrun_list = (clusterizer.read_subruns())
        subruns_to_do = (set(subrun_list) - set(done_subruns))
        subrun_list = [x for x in subruns_to_do if x <= subrun_tgt]
        sub_list=[]
        sub_data = clusterizer.data_pd.groupby("subRunNo")
        for key in sub_data.groups:
            if key in sub_list:
                sub_list.append(sub_data.get_group(key))
        del data_pd
        del clusterizer.data_pd

        if not self.silent:
            print ("Single view")
        if len(sub_list)>0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(sub_list), disable=self.silent) as pbar:
                    for i, x in enumerate(pool.imap_unordered(clusterizer.build_view_clusters, sub_list)):
                        pd_1d_return_list.append(x)
                        pbar.update()
            clusterizer.cluster_pd=pd.concat(pd_1d_return_list)
        else:
            print ("No subrun to clusterize, is the file hit_data.pickle.gzip in the working folder? Try to launch with -a")
            return (1)
        clusterizer.append_cluster_pd()

        if not self.cylinder:

            if not self.silent:
                print ("Clusters 2-D")
            if len(subrun_list) > 0:
                with Pool(processes=self.cpu_to_use) as pool:
                    with tqdm(total=len(subrun_list), disable=self.silent) as pbar_2:
                        for i, x in enumerate(pool.imap_unordered(clusterizer.build_2D_clusters, subrun_list)):
                            pd_2d_return_list.append(x)
                            pbar_2.update()
                clusterizer.cluster_pd_2D = pd.concat(pd_2d_return_list)

            clusterizer.append_cluster_pd_2D()

    def clusterize_subrun(self,subrun, time_limits):
        """
        Clusterize one run
        :param run_number:
        :param data_folder:
        :return:
        """
        if time_limits:
            clusterizer = pl_lib.clusterize(self.run_number, self.data_folder, time_limits[0], time_limits[1])
        else:
            clusterizer = pl_lib.clusterize.default_time_winw(self.run_number, self.data_folder)
        clusterizer.load_data_pd()
        subrun_list = (clusterizer.read_subruns())
        if subrun in subrun_list:
            sub_data = clusterizer.data_pd.groupby("subRunNo")
            sub_data_to_do = sub_data.get_group(subrun)
            clusterizer.cluster_pd=clusterizer.build_view_clusters(sub_data_to_do)
        else:
            print(f"Can't find subrun {subrun} to clusterize, is the file hit_data.pickle.gzip in the working folder? Try to launch with -a")
            return (1)
        clusterizer.save_cluster_pd(subrun)
        clusterizer.cluster_pd_2D=clusterizer.build_2D_clusters(subrun)
        clusterizer.save_cluster_pd_2D(subrun)

### tracking part ###

    def tracks2d_run(self):
        """
        Build tracks for one run one run
        :param run_number:
        :param data_folder:
        :return:
        """
        tracking_return_list=[]
        tracker=pl_lib.tracking_2d(self.run_number, self.data_folder)
        tracker.load_cluster_2D()
        subrun_list=(tracker.read_subruns())
        if not self.silent:
            print ("Tracking")
        if len(subrun_list)>0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(subrun_list), disable=self.silent) as pbar:
                    for i, x in enumerate(pool.imap_unordered(tracker.build_tracks_pd, subrun_list)):
                        tracking_return_list.append(x)
                        pbar.update()
            tracker.tracks_pd=pd.concat(tracking_return_list)
        else:
            print ("No subrun to clusterize, is the file hit_data.pickle.gzip in the working folder? Try to launch with -a")
            return (1)
        tracker.save_tracks_pd()


    def tracks2D_run_fill(self, subrun_tgt):
        """
        Clusterize one run
        :param run_number:
        :param data_folder:
        :return:
        """
        tracking_return_list = []
        tracker = pl_lib.tracking_2d(self.run_number, self.data_folder)
        tracker.load_cluster_2D()
        if not self.silent:
            print(f"Tracking filling up to subrun {subrun_tgt}")

        path = self.data_folder + f"/raw_root/{self.run_number}/tracks_pd.pickle.gzip"
        if os.path.isfile(path):
            data_pd = pd.read_pickle(path, compression="gzip")
            done_subruns = data_pd.subrun.unique()
        else:
            done_subruns = []

        subrun_list = (tracker.read_subruns())
        subruns_to_do = (set(subrun_list) - set(done_subruns))
        subrun_list = [x for x in subruns_to_do if x <= subrun_tgt]
        if not self.silent:
            print ("Single view")
        if len(subrun_list)>0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(subrun_list), disable=self.silent) as pbar:
                    for i, x in enumerate(pool.imap_unordered(tracker.build_tracks_pd, subrun_list)):
                        tracking_return_list.append(x)
                        pbar.update()
            tracker.tracks_pd=pd.concat(tracking_return_list)
            tracker.append_tracks_pd()

        else:
            print ("No subrun to clusterize, is the file hit_data.pickle.gzip in the working folder? Try to launch with -a")
            return (1)


    def track2D_subrun(self,subrun):
        """
        Clusterize one run
        :param run_number:
        :param data_folder:
        :return:
        """

        tracker=pl_lib.tracking_2d(self.run_number, self.data_folder)
        tracker.load_cluster_2D()
        subrun_list=(tracker.read_subruns())

        if subrun in subrun_list:
            tracker.tracks_pd=tracker.build_tracks_pd(subrun)
        else:
            print(f"Can't find subrun {subrun} to track, is the file cluster file in place? try to run with -c")
            return (1)
        tracker.save_tracks_pd(subrun)

### tracking 1D part ###

    def tracks_run(self):
        """
        Build tracks for one run one run
        :param run_number:
        :param data_folder:
        :return:
        """

        tracking_return_list = []
        tracker = pl_lib.tracking_1d(self.run_number, self.data_folder, self.alignment, self.cylinder)
        tracker.load_cluster_1D(self.cylinder)
        if not self.silent:
            print ("Preparing data")

        sub_list=[]
        sub_data = tracker.cluster_pd_1D.groupby("subrun")
        for key in sub_data.groups:
            subrun_pd=sub_data.get_group(key)

            if self.cylinder:
                subrun_pd = subrun_pd.apply(pl_lib.change_planar, 1)
            sub_list.append(subrun_pd)

        del tracker.cluster_pd_1D
        if not self.silent:
            print ("Single view tracking")
        if len(sub_list)>0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(sub_list), disable=self.silent) as pbar:
                    for i, x in enumerate(pool.imap_unordered(tracker.build_tracks_pd, sub_list)):
                        tracking_return_list.append(x)
                        pbar.update()
            tracker.tracks_pd=pd.concat(tracking_return_list)
            tracker.save_tracks_pd()
        else:
            print ("No subrun to fit, is the file cluster_pd_1D.pickle.gzip in the working folder? Try to launch with -c")
            return (1)


    def tracks_run_fill(self, subrun_tgt):
        """
        Clusterize one run
        :param run_number:
        :param data_folder:
        :return:
        """

        if not self.silent:
            print(f"Tracking filling up to subrun {subrun_tgt}")
        tracking_return_list = []
        tracker = pl_lib.tracking_1d(self.run_number, self.data_folder, self.alignment, self.cylinder)
        tracker.load_cluster_1D(self.cylinder)
        path = self.data_folder + f"/raw_root/{self.run_number}/tracks_pd_1D.pickle.gzip"
        if not self.silent:
            print ("Preparing data")
        if os.path.isfile(path):
            data_pd = pd.read_pickle(path, compression="gzip")
            done_subruns = data_pd.subrun.unique()
        else:
            done_subruns = []

        subrun_list = (tracker.read_subruns())
        subruns_to_do = (set(subrun_list) - set(done_subruns))
        subrun_list = [x for x in subruns_to_do if x <= subrun_tgt]


        sub_list=[]
        sub_data = tracker.cluster_pd_1D.groupby("subrun")
        for key in sub_data.groups:
            if key in subrun_list:
                subrun_pd=sub_data.get_group(key)

                if self.cylinder:
                    subrun_pd = subrun_pd.apply(pl_lib.change_planar, 1)
                sub_list.append(subrun_pd)

        del tracker.cluster_pd_1D
        if not self.silent:
            print ("Single view tracking")
        if len(sub_list)>0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(sub_list), disable=self.silent) as pbar:
                    for i, x in enumerate(pool.imap_unordered(tracker.build_tracks_pd, sub_list)):
                        tracking_return_list.append(x)
                        pbar.update()
            tracker.tracks_pd=pd.concat(tracking_return_list)
            tracker.append_tracks_pd()
        else:
            print ("No subrun to fit, is the file cluster_pd_1D.pickle.gzip in the working folder? Try to launch with -c")
            return (1)


    def track_subrun(self,subrun):
        """
        Clusterize one run
        :param run_number:
        :param data_folder:
        :return:
        """

        tracker= pl_lib.tracking_1d(self.run_number, self.data_folder, self.alignment, self.cylinder)
        tracker.load_cluster_1D()
        subrun_list=(tracker.read_subruns())

        if subrun in subrun_list:
            tracker.tracks_pd=tracker.build_tracks_pd(subrun)
        else:
            print(f"Can't find subrun {subrun} to track, is the file cluster file in place? try to run with -c")
            return (1)
        tracker.save_tracks_pd(subrun)

### selection 1D part ###

    def select_run(self):
        """
        Build tracks for one run one run
        :param run_number:
        :param data_folder:
        :return:
        """
        tracking_return_list = []

        tracker = pl_lib.tracking_1d(self.run_number, self.data_folder, self.alignment, self.cylinder)
        if not self.silent:
            print("Loading clusters")
        tracker.load_cluster_1D()
        if not self.silent:
            print("Loading tracks")
        tracker.load_tracks_pd()
        sub_list=[]
        tracks_sub_list=[]
        if not self.silent:
            print("Preparing data")
        sub_data = tracker.cluster_pd_1D.groupby("subrun")
        sub_data_tracks = tracker.tracks_pd.groupby("subrun")
        keys_tracker=sub_data_tracks.groups
        for key in sub_data.groups:
            if key in keys_tracker:
                subrun_pd=sub_data.get_group(key)
                if self.cylinder:
                    subrun_pd = subrun_pd.apply(pl_lib.change_planar, 1)
                sub_list.append(subrun_pd)
                tracks_sub_list.append(sub_data_tracks.get_group(key))
        del tracker.cluster_pd_1D
        del tracker.tracks_pd
        input_list=list(zip(sub_list,tracks_sub_list))
        if not self.silent:
            print("Selcting cluster using tracks")
        if len(input_list) > 0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(input_list), disable=self.silent) as pbar:
                    for i, x in enumerate(pool.imap_unordered(tracker.build_select_cl_pd, input_list)):
                        tracking_return_list.append(x)
                        pbar.update()
        else:
            print("No track information to select subruns")
            return (1)
        if not self.silent:
            print("Saving")
        tracker.cluster_pd_1D_selected = pd.concat(tracking_return_list, ignore_index=True)
        tracker.save_sel_cl_pd()

    def select_run_fill(self, subrun_tgt):
        """
        Clusterize one run
        :param run_number:
        :param data_folder:
        :return:
        """
        tracking_return_list = []
        tracker = pl_lib.tracking_1d(self.run_number, self.data_folder, self.alignment, self.cylinder)


        if not self.silent:
            print(f"Selecting clusters near tracks subrun {subrun_tgt}")

        path = self.data_folder + f"/raw_root/{self.run_number}/sel_cluster_pd_1D.pickle.gzip"
        if os.path.isfile(path):
            data_pd = pd.read_pickle(path, compression="gzip")
            done_subruns = data_pd.subrun.unique()
        else:
            done_subruns = []

        subrun_list = (tracker.read_subruns(True))
        subruns_to_do = (set(subrun_list) - set(done_subruns))
        subrun_list = [x for x in subruns_to_do if x <= subrun_tgt]

        if not self.silent:
            print("Loading clusters")
        tracker.load_cluster_1D()
        if not self.silent:
            print("Loading tracks")
        tracker.load_tracks_pd()
        sub_list=[]
        tracks_sub_list=[]
        if not self.silent:
            print("Preparing data")
        sub_data = tracker.cluster_pd_1D.groupby("subrun")
        sub_data_tracks = tracker.tracks_pd.groupby("subrun")
        keys_tracker=sub_data_tracks.groups
        for key in sub_data.groups:
            if key in keys_tracker and key in subrun_list:
                subrun_pd = sub_data.get_group(key)
                if self.cylinder:
                    subrun_pd = subrun_pd.apply(pl_lib.change_planar, 1)
                sub_list.append(subrun_pd)
                tracks_sub_list.append(sub_data_tracks.get_group(key))
        del tracker.cluster_pd_1D
        del tracker.tracks_pd
        input_list = list(zip(sub_list, tracks_sub_list))
        if not self.silent:
            print("Selcting cluster using tracks")
        if len(input_list) > 0:
            with Pool(processes=self.cpu_to_use) as pool:
                with tqdm(total=len(input_list), disable=self.silent) as pbar:
                    for i, x in enumerate(pool.imap_unordered(tracker.build_select_cl_pd, input_list)):
                        tracking_return_list.append(x)
                        pbar.update()
        else:
            print("No track information to select subruns")
            return (1)
        if not self.silent:
            print("Saving")
        tracker.cluster_pd_1D_selected = pd.concat(tracking_return_list, ignore_index=True)
        tracker.append_sel_cl_pd()


    def select_subrun(self, subrun):
        """
        Clusterize one run
        :param run_number:
        :param data_folder:
        :return:
        """

        tracker = pl_lib.tracking_1d(self.run_number, self.data_folder, self.alignment, self.cylinder)
        tracker.load_cluster_1D()

        subrun_list = (tracker.read_subruns(True))

        if subrun in subrun_list:
            tracker.tracks_pd = tracker.build_select_cl_pd(subrun)
        else:
            print(f"Can't find subrun {subrun} to track, is the file cluster file in place? try to run with -c")
            return (1)
        tracker.save_sel_cl_pd(subrun)


    def save_config(self,args):
        """
        Saves the current configuration in a pickle file.
        """
        conf_path = self.data_folder+"/raw_root/"+str(self.run_number)+"/analisis_config"
        if os.path.isfile(conf_path):
            os.rename(conf_path, conf_path+"_old")
        with open (conf_path, 'w') as conf_file:
            (args.__dict__.pop("method"))
            json.dump(args.__dict__,conf_file,  indent=2)

    def calibrate_alignment(self):
        """
        Calbrate the aligment matrix for performance evaluation
        :return:
        """
        import alignment
        alignment.calibrate_alignment_run(run = self.run_number, rounds = 3, cpu = self.cpu_to_use, data_folder=self.data_folder, downsampling=self.downsampling)

    def compress_hit_pd(self):
        """
        Optimize the disk usage for the hit pd
        :return:
        """
        clusterizer = pl_lib.clusterize.default_time_winw(self.run_number, self.data_folder)
        clusterizer.load_data_pd()
        compress_data = pl_lib.compress_hit_pd(clusterizer.data_pd)
        pl_lib.verifiy_compression_validity(clusterizer.data_pd, compress_data)
        compress_data.to_pickle("{}/raw_root/{}/hit_data.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")
    def convert_hit_pd_root(self):
        """
        Convert the hit file in root
        :return:
        """
        clusterizer = pl_lib.clusterize.default_time_winw(self.run_number, self.data_folder)
        clusterizer.load_data_pd()
        import ROOT as R
        #Convert PD in dict of arrays
        data = {key: np.array(clusterizer.data_pd[key].values) for key in clusterizer.data_pd.columns}
        #Fix data types for root import
        data['channel'] = data['channel'].astype(np.int32)
        data['tac'] = data['tac'].astype(np.int32)
        data['tcoarse'] = data['tcoarse'].astype(np.int32)
        data['ecoarse'] = data['ecoarse'].astype(np.int32)
        data['tfine'] = data['tfine'].astype(np.int32)
        data['efine'] = data['efine'].astype(np.int32)
        data['tiger'] = data['tiger'].astype(np.int32)
        data['l1ts_min_tcoarse'] = data['l1ts_min_tcoarse'].astype(np.int32)
        data['delta_coarse'] = data['delta_coarse'].astype(np.int32)
        data['timestamp'] = data['timestamp'].astype(np.int32)
        data['gemroc'] = data['gemroc'].astype(np.int32)
        data['runNo'] = data['runNo'].astype(np.int32)
        data['subRunNo'] = data['subRunNo'].astype(np.int32)
        data['l1_framenum'] = data['l1_framenum'].astype(np.int32)
        data['hit_id'] = data['hit_id'].astype(np.int32)
        data['strip_x'] = data['strip_x'].astype(np.int32)
        data['strip_v'] = data['strip_y'].astype(np.int32)
        del data["strip_y"]
        data['layer'] = data['planar'].astype(np.int32)
        del data['planar']
        data['FEB_label'] = data['FEB_label'].astype(np.int32)
        data['charge_SH'] = data['charge_SH'].astype(np.float32)
        rdf = R.RDF.MakeNumpyDataFrame(data)
        folder= os.path.join(self.data_folder_root, str(self.run_number))
        if not os.path.isdir(folder):
            os.mkdir(folder)
        rdf.Snapshot('tree', os.path.join(self.data_folder_root,  str(self.run_number),"ana.root" ))

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
    if args.cylinder:
        config_file="config_cyl.ini"
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

    if (args.data_folder_root):
        data_folder_root=args.data_folder_root

    if (args.mapping):
        mapping_file=args.mapping
    if args.subrun==-1:
        subrun_tgt= "All"
    else:
        subrun_tgt=int(args.subrun)

    subrun_fill=args.subrun_fill
    if not args.Silent:
        print ("#############################################################")
        print (f"Run : {run}")
        print (f"Data_folder : {data_folder}")
        print (f"Calib_folder : {calib_folder}")
        print (f"mapping_file : {mapping_file}")
        print (f"Subrun : {subrun_tgt}")
        if args.downsampling:
            print (f"Downsamplig : 1:{args.downsampling}")
        if args.cylinder:
            print ("Geometry: Cylinder")
        else:
            print ("Geometry: Planar")
        print (f"Operations:")
        if args.decode:
            print ("        -Decode")
        if args.ana:
            print ("        -Analyze")
        if args.clusterize:
            print ("        -Clusterize")
        if args.alignment:
            print("---- Using Alignment ----")
        if args.calibrate_alignment:
            print ("-Calibrating alignemnt")
        if args.compress:
            print ("-Compressing")
        if args.tracking:
            print ("        -Tracking")
        if args.selection:
            print("         -Selection")
        if args.root_conv:
            print("         -Converting to root")
        if args.performance:
            print("         -Performance evaluation")
        # if not (args.decode | args.ana | args.clusterize | args.tracking | args.selection | args.calibrate_alignment | args.compress | args.root_conv |args.performance):
        #     print ("        -Decode\n        -Analyze\n        -Clusterize\n        -Tracking \n        -Selection")
        if args.cpu:
            print (f"Parallel on {args.cpu} CPUs")
        print ("#############################################################")

    op_list=[]
    if args.decode:
        op_list.append("D")
    if args.ana:
        op_list.append("A")

    if args.clusterize:
        op_list.append("C")

    if args.tracking:
        op_list.append("T")

    if args.selection:
        op_list.append("S")

    if args.calibrate_alignment:
        op_list.append("c_align")
    if args.compress:
        op_list.append("compress")
    if args.root_conv:
        op_list.append("root_conv")
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
    if args.ir:
        options["purge"]=False
    if args.alignment:
        options["alignment"]=True
    else:
        options["alignment"]=False
    if args.cylinder:
        options["cylinder"]=True
    if args.downsampling:
        options["downsampling"] = args.downsampling
    if args.data_folder_root:
        options["data_folder_root"] = args.data_folder_root
    if len (op_list)>0:
        main_runner = runner(data_folder,run,calib_folder,mapping_file,**options)
    else:
        sys.exit(0)

    if "D" in (op_list):
        if not os.path.isdir(f"{data_folder}/raw_root/{run}"):
            os.mkdir(f"{data_folder}/raw_root/{run}")
        if subrun_fill == -1:
            if subrun_tgt== "All":
                main_runner.dec_run()
                main_runner.merge_dec()

            else:
                main_runner.dec_subrun(subrun_tgt)
                main_runner.merge_dec()
        else:
            main_runner.dec_run_fill(subrun_fill)
            main_runner.merge_dec()

    if "A" in (op_list):
        if not args.Silent:
            print ("Calibrating and mapping")
        if subrun_fill == -1:
            if subrun_tgt == "All":
                main_runner.calib_run()
            else:
                main_runner.calib_subrun(subrun_tgt)
        else:
            main_runner.calib_run_fill(subrun_fill)

    if "root_conv" in (op_list):
        main_runner.convert_hit_pd_root()

    if "C" in (op_list):
        if not args.Silent:
            print ("Clusterizing")
        if subrun_fill == -1:
            if subrun_tgt == "All":
                main_runner.clusterize_run(args.time_window)
            else:
                main_runner.clusterize_subrun(subrun_tgt, args.time_window)
        else:
            main_runner.clusterize_run_fill(subrun_fill, args.time_window)

    if "T" in (op_list):
        if subrun_fill == -1:
            if subrun_tgt == "All":
                main_runner.tracks_run()
            else:
                main_runner.track_subrun(subrun_tgt)
        else:
            main_runner.tracks_run_fill(subrun_fill)

    if "S" in (op_list):
        if subrun_fill == -1:
            if subrun_tgt == "All":
                main_runner.select_run()
            else:
                main_runner.select_subrun(subrun_tgt)
        else:
            main_runner.select_run_fill(subrun_fill)
    if "c_align" in (op_list):
        main_runner.calibrate_alignment()
    if "compress" in (op_list):
        main_runner.compress_hit_pd()

    if "perf" in (op_list):

        main_runner.eval_perf(args.performance)
    main_runner.save_config(args)



if __name__=="__main__":

    parser = argparse.ArgumentParser(description="CIVETTA: Tools to decode and analyze TIGER data",)
    parser.add_argument('run', type=int, help='Run number')
    parser.set_defaults(method=main)
    parser.add_argument('-d','--decode',  action="store_true", help='Decode')
    parser.add_argument('-a','--ana', help='Analyze (=calibrate and apply mapping)', action="store_true")
    parser.add_argument('-c','--clusterize', help='clusterize ', action="store_true")
    parser.add_argument('-t','--tracking', help='Tracks building (for data selection)', action="store_true")
    parser.add_argument('-sel','--selection', help='Selects 1-D clusters based on tracks', action="store_true")

    parser.add_argument('-df','--data_folder', type=str, required=False,
                        help='Specify the data folder, default set by the .ini file')
    parser.add_argument('-m','--mapping', type=str, required=False,
                        help='Specify the mapping file, default set by the .ini file')
    parser.add_argument('-s','--subrun', help='Run on only one subrun ', type=int, default=-1)
    parser.add_argument('-S','--Silent', help='Print only errors ',action="store_true")
    parser.add_argument('-cpu','--cpu', help='Specify CPU count ',type=int)
    parser.add_argument('-ir','--ir', help='Inhibit the removal of process files ',action="store_true")
    parser.add_argument('-tw','--time_window', help='Specify the signal time window for clusterization', type=int, nargs=2)
    parser.add_argument('-sf','--subrun_fill', help='Runs to fill up to the subrun', type=int, default=-1)
    parser.add_argument('-ali','--alignment', help='Use the alignment', action="store_true")
    parser.add_argument('-root','--root_conv', help='Convert hit_pd in root', action="store_true")
    parser.add_argument('-rootf','--data_folder_root', help='Specify the folder for the root files (without run folder)', type=str, required=False)

    parser.add_argument('-down','--downsampling', help='Downsample the decoded data to speed up analysis ',type=int)
    parser.add_argument('-cyl','--cylinder', help='Cylindrical geometry ', action="store_true")
    parser.add_argument('-ca_al','--calibrate_alignment', help='Calibrate the alignemnt on the file ', action="store_true")
    parser.add_argument('-comp','--compress', help='Optimize the disk usage ', action="store_true")
    parser.add_argument('-perf','--performance', help='Performance evaluation ', type=int)

    args = parser.parse_args()
    args.method(**vars(args))
