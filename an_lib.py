import ROOT as R
import root_numpy
import glob2
import numpy as np
import root_pandas
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm_notebook
from sklearn.cluster import KMeans
from math import log10
from scipy.stats import moyal
from scipy.optimize import curve_fit
import pickle


def extract_num_triggers_subrun(filename):
    with open(filename, 'r') as log_file:
        for line in log_file.readlines():
            if "total packets" in line:
                return line.split(" ")[-1]
    return 0
class integ_time_analyze:
    def __init__(self, run_list, data_folder, out_folder):
        self.run_list = run_list
        self.data_folder = data_folder
        self.out_folder = out_folder
        try:
            self.buid_cluster_pd_list_sel()
        except:
            print ("Can't use selected clusters")
            self.build_cluster_pd_list()
        self.build_cluster_pd_2D_list()

    def build_cluster_pd_list(self):
        cluster_pd_list = []
        for run_number in self.run_list:
            cluster_pd_list.append(pd.read_pickle("{}/raw_root/{}/cluster_pd_1D.pickle.gzip".format(self.data_folder, run_number), compression="gzip"))
        self.cluster_pd_list = cluster_pd_list

    def build_cluster_pd_2D_list(self):
        cluster_pd_2D_list = []
        for run_number in self.run_list:
            cluster_pd_2D_list.append(pd.read_pickle("{}/raw_root/{}/cluster_pd_2D.pickle.gzip".format(self.data_folder, run_number), compression="gzip"))
        self.cluster_pd_2D_list=cluster_pd_2D_list

    def buid_cluster_pd_list_sel(self):
        cluster_pd_list = []
        for run_number in self.run_list:
            cluster_pd_list.append(pd.read_pickle("{}/raw_root/{}/sel_cluster_pd_1D.pickle.gzip".format(self.data_folder, run_number), compression="gzip"))
        self.cluster_pd_list=cluster_pd_list

    def extract_num_triggers_run(self,run_number):
        tot=0
        for filename, subrun in glob2.iglob("/{}/raw_dat/RUN_{}/ACQ_log_*".format(self.data_folder,run_number),with_matches=True):
            tot+=int(extract_num_triggers_subrun(filename))
        return tot

    def calculate_y_landau(self,fit, nbins, range_b):
        y_data=[]
        for x in np.arange (0,range_b,range_b/nbins):
            y_data.append(R.TMath.Landau(x,fit[1],fit[2])*fit[0])
        return y_data

    def get_min_max_integ(self,run):
        with open (f"{self.data_folder}/raw_dat/RUN_{run}/CONF_run_{run}.pkl", 'rb') as conf_dict_f_op:
            conf_dict=pickle.load(conf_dict_f_op)
        return(conf_dict["GEMROC 1"]["TIGER 1"]["Ch 10"]["MaxIntegTime"])

    def fill_hist_and_norm_1(self,cluster_pd, nbins=600, range_b=300):
        h=R.TH1F(f"h_{cluster_pd.run.unique()[0]}",f"h_{cluster_pd.run.unique}", nbins, 0, range_b)
        for x in cluster_pd.cl_charge:
            h.Fill(x)
        h.Scale(1/h.Integral())
        return h

    def fill_hist_and_norm_1_view(self,cluster_pd, nbins=600, range_b=300, view="x"):
        h= R.TH1F(f"h_{cluster_pd.run.unique()[0]}",f"h_{cluster_pd.run.unique}", nbins, 0, range_b)
        for x in cluster_pd[f"cl_charge_{view}"]:
            h.Fill(x)
        h.Scale(1/h.Integral())
        return h


    def analize_1_planar_1D_view(self, planar,  view):
        nbins = 1000
        range_b = 300
        root_hist_list = []
        for cluster_pd, run in zip(self.cluster_pd_list, self.run_list):
            df_c=cluster_pd[(cluster_pd.planar == planar) & (cluster_pd[f"cl_pos_{view}"] > 0)]
            h_n = self.fill_hist_and_norm_1(df_c, nbins=nbins, range_b=range_b)
            root_hist_list.append(h_n)
        fit_list = []
        error_list = []
        for n, histo in enumerate(root_hist_list):
            histo.SetLineColor(n + 1)
            histo.Fit("landau", "ww S")
            histo.GetFunction("landau").SetLineColor(n + 1)
            fit_list.append(histo.GetFunction("landau").GetParameters())
            error_list.append(histo.GetFunction("landau").GetParError(1))
        lines_list = []
        for fit, run in zip(fit_list, self.run_list):
            y_data = self.calculate_y_landau(fit, nbins=nbins, range_b=range_b)
            lines_list.append(go.Scatter(x=np.arange(0, range_b, range_b / nbins), y=y_data, name=f"Fit run {run}"))
        hist_list = []
        for cluster_pd, run in zip(self.cluster_pd_list, self.run_list):
            df_c=cluster_pd[(cluster_pd.planar == planar) & (cluster_pd[f"cl_pos_{view}"] > 0)]

            hist_list.append(go.Histogram(x=df_c.cl_charge,
                                          histnorm="probability",
                                          name=f"Run {run}", opacity=0.75,
                                          xbins=dict(size=range_b / nbins, start=0)))
        fig = go.Figure(hist_list + lines_list)
        fig.update_layout(title="Cluster charge selected by track",
                          xaxis_title="Cluster charge [fC]",
                          bargroupgap=0,
                          barmode='overlay',

                          yaxis_title="#")
        fig.show()
        avg_list = []
        error_avg_list=[]
        tot_list=[]
        for cluster_pd, run in zip(self.cluster_pd_list, self.run_list):
            df_c=cluster_pd[(cluster_pd.planar == planar) & (cluster_pd[f"cl_pos_{view}"] > 0)]
            avg_list.append(np.average(df_c.cl_charge))
            error_avg_list.append(
                np.std(df_c.cl_charge) /
                (df_c.cl_charge.count()) ** (1 / 2)
            )
            tot_list.append(df_c.cl_charge.count())
        min_max_list = []
        MPV_list = []
        for run, FIT in zip(self.run_list, fit_list):
            MPV_list.append(FIT[1])
            min_max_list.append(self.get_min_max_integ(run))
            # print(f"Avg charge cl 1-D {view} run {run}, integ time {self.get_min_max_integ(run)}: {FIT[1]}")
        d = {'run': list(map(str, self.run_list)), 'MPV': MPV_list, "Integ_time": min_max_list, "avg": avg_list, "MPV_error": error_list, "AVG_error": error_avg_list, "Tot_cluster": tot_list}
        result_pd = pd.DataFrame(data=d)

        c=R.TCanvas()
        root_hist_list[0].Draw("histo")
        for h in root_hist_list[1:]:
            h.Draw("histo same")
        c.Draw()
        # c.Print(f"{self.out_folder}/fit_root_1d_pl{planar}_view{view}.pdf")
        return result_pd


    def analize_all_planar_1D_view(self,   view):
        nbins = 1000
        range_b = 300
        root_hist_list = []
        for cluster_pd, run in zip(self.cluster_pd_list, self.run_list):
            df_c=cluster_pd[ (cluster_pd[f"cl_pos_{view}"] > 0)]
            h_n = self.fill_hist_and_norm_1(df_c, nbins=nbins, range_b=range_b)
            root_hist_list.append(h_n)
        fit_list = []
        error_list = []
        for n, histo in enumerate(root_hist_list):
            histo.SetLineColor(n + 1)
            histo.Fit("landau", "ww S")
            histo.GetFunction("landau").SetLineColor(n + 1)
            fit_list.append(histo.GetFunction("landau").GetParameters())
            error_list.append(histo.GetFunction("landau").GetParError(1))
        lines_list = []
        for fit, run in zip(fit_list, self.run_list):
            y_data = self.calculate_y_landau(fit, nbins=nbins, range_b=range_b)
            lines_list.append(go.Scatter(x=np.arange(0, range_b, range_b / nbins), y=y_data, name=f"Fit run {run}"))
        hist_list = []
        for cluster_pd, run in zip(self.cluster_pd_list, self.run_list):
            df_c=cluster_pd[ (cluster_pd[f"cl_pos_{view}"] > 0)]

            hist_list.append(go.Histogram(x=df_c.cl_charge,
                                          histnorm="probability",
                                          name=f"Run {run}", opacity=0.75,
                                          xbins=dict(size=range_b / nbins, start=0)))
        fig = go.Figure(hist_list + lines_list)
        fig.update_layout(title="Cluster charge selected by track",
                          xaxis_title="Cluster charge [fC]",
                          bargroupgap=0,
                          barmode='overlay',

                          yaxis_title="#")
        fig.show()
        avg_list = []
        error_avg_list=[]
        tot_list=[]
        for cluster_pd, run in zip(self.cluster_pd_list, self.run_list):
            df_c=cluster_pd[ (cluster_pd[f"cl_pos_{view}"] > 0)]
            avg_list.append(np.average(df_c.cl_charge))
            error_avg_list.append(
                np.std(df_c.cl_charge) /
                (df_c.cl_charge.count()) ** (1 / 2)
            )
            tot_list.append(df_c.cl_charge.count())
        min_max_list = []
        MPV_list = []
        for run, FIT in zip(self.run_list, fit_list):
            MPV_list.append(FIT[1])
            min_max_list.append(self.get_min_max_integ(run))
            # print(f"Avg charge cl 1-D {view} run {run}, integ time {self.get_min_max_integ(run)}: {FIT[1]}")
        d = {'run': list(map(str, self.run_list)), 'MPV': MPV_list, "Integ_time": min_max_list, "avg": avg_list, "MPV_error": error_list, "AVG_error": error_avg_list, "Tot_cluster": tot_list}
        result_pd = pd.DataFrame(data=d)

        c=R.TCanvas()
        root_hist_list[0].Draw("histo")
        for h in root_hist_list[1:]:
            h.Draw("histo same")
        c.Draw()
        # c.Print(f"{self.out_folder}/fit_root_1d_pl{planar}_view{view}.pdf")
        return result_pd

    def analize_1_planar_2D_view(self, planar, view):
        nbins = 1000
        range_b = 300
        root_hist_list = []
        for cluster_pd, run in zip(self.cluster_pd_2D_list, self.run_list):
            df_c = cluster_pd[(cluster_pd.planar == planar) & (cluster_pd[f"cl_pos_{view}"] > 0)]
            h_n = self.fill_hist_and_norm_1_view(df_c, nbins=nbins, range_b=range_b)
            root_hist_list.append(h_n)
        fit_list = []
        error_list = []
        for n, histo in enumerate(root_hist_list):
            histo.SetLineColor(n + 1)
            histo.Fit("landau", "ww S")
            histo.GetFunction("landau").SetLineColor(n + 1)
            fit_list.append(histo.GetFunction("landau").GetParameters())
            error_list.append(histo.GetFunction("landau").GetParError(1))
        lines_list = []
        for fit, run in zip(fit_list, self.run_list):
            y_data = self.calculate_y_landau(fit, nbins=nbins, range_b=range_b)
            lines_list.append(go.Scatter(x=np.arange(0, range_b, range_b / nbins), y=y_data, name=f"Fit run {run}"))
        hist_list = []
        for cluster_pd, run in zip(self.cluster_pd_2D_list, self.run_list):
            df_c = cluster_pd[(cluster_pd.planar == planar) & (cluster_pd[f"cl_pos_{view}"] > 0)]

            hist_list.append(go.Histogram(x=df_c[f"cl_charge_{view}"],
                                          histnorm="probability",
                                          name=f"Run {run}", opacity=0.75,
                                          xbins=dict(size=range_b / nbins, start=0)))
        fig = go.Figure(hist_list + lines_list)
        fig.update_layout(title="Cluster charge 2D",
                          xaxis_title="Cluster charge [fC]",
                          bargroupgap=0,
                          barmode='overlay',

                          yaxis_title="#")
        fig.show()
        avg_list = []
        error_avg_list = []
        tot_list = []
        for cluster_pd, run in zip(self.cluster_pd_2D_list, self.run_list):
            df_c = cluster_pd[(cluster_pd.planar == planar) & (cluster_pd[f"cl_pos_{view}"] > 0)]
            avg_list.append(np.average(df_c[f"cl_charge_{view}"]))
            error_avg_list.append(
                np.std(df_c[f"cl_charge_{view}"]) /
                (df_c[f"cl_charge_{view}"].count()) ** (1 / 2)
            )
            tot_list.append(df_c[f"cl_charge_{view}"].count())
        min_max_list = []
        MPV_list = []
        for run, FIT in zip(self.run_list, fit_list):
            MPV_list.append(FIT[1])
            min_max_list.append(self.get_min_max_integ(run))
            # print(f"Avg charge cl 1-D {view} run {run}, integ time {self.get_min_max_integ(run)}: {FIT[1]}")
        d = {'run': list(map(str, self.run_list)), 'MPV': MPV_list, "Integ_time": min_max_list, "avg": avg_list, "MPV_error": error_list, "AVG_error": error_avg_list, "Tot_cluster": tot_list}
        result_pd = pd.DataFrame(data=d)

        c = R.TCanvas()
        root_hist_list[0].Draw("histo")
        for h in root_hist_list[1:]:
            h.Draw("histo same")
        c.Draw()
        # c.Print(f"{self.out_folder}/fit_root_1d_pl{planar}_view{view}.pdf")
        return result_pd

    def analize_1_planar_2D(self, planar):
        nbins = 1000
        range_b = 300
        root_hist_list = []
        for cluster_pd, run in zip(self.cluster_pd_2D_list, self.run_list):
            df_c = cluster_pd[(cluster_pd.planar == planar) ]
            h_n = self.fill_hist_and_norm_1(df_c, nbins=nbins, range_b=range_b)
            root_hist_list.append(h_n)
        fit_list = []
        error_list = []
        for n, histo in enumerate(root_hist_list):
            histo.SetLineColor(n + 1)
            histo.Fit("landau", "ww S")
            histo.GetFunction("landau").SetLineColor(n + 1)
            fit_list.append(histo.GetFunction("landau").GetParameters())
            error_list.append(histo.GetFunction("landau").GetParError(1))
        lines_list = []
        for fit, run in zip(fit_list, self.run_list):
            y_data = self.calculate_y_landau(fit, nbins=nbins, range_b=range_b)
            lines_list.append(go.Scatter(x=np.arange(0, range_b, range_b / nbins), y=y_data, name=f"Fit run {run}"))
        hist_list = []
        for cluster_pd, run in zip(self.cluster_pd_2D_list, self.run_list):
            df_c = cluster_pd[(cluster_pd.planar == planar) ]

            hist_list.append(go.Histogram(x=df_c.cl_charge,
                                          histnorm="probability",
                                          name=f"Run {run}", opacity=0.75,
                                          xbins=dict(size=range_b / nbins, start=0)))
        fig = go.Figure(hist_list + lines_list)
        fig.update_layout(title="Cluster charge 2D",
                          xaxis_title="Cluster charge [fC]",
                          bargroupgap=0,
                          barmode='overlay',

                          yaxis_title="#")
        fig.show()
        avg_list = []
        error_avg_list = []
        tot_list = []
        for cluster_pd, run in zip(self.cluster_pd_2D_list, self.run_list):
            df_c = cluster_pd[(cluster_pd.planar == planar) ]
            avg_list.append(np.average(df_c.cl_charge))
            error_avg_list.append(
                np.std(df_c.cl_charge) /
                (df_c.cl_charge.count()) ** (1 / 2)
            )
            tot_list.append(df_c.cl_charge.count())
        min_max_list = []
        MPV_list = []
        for run, FIT in zip(self.run_list, fit_list):
            MPV_list.append(FIT[1])
            min_max_list.append(self.get_min_max_integ(run))
            # print(f"Avg charge cl 1-D {view} run {run}, integ time {self.get_min_max_integ(run)}: {FIT[1]}")
        d = {'run': list(map(str, self.run_list)), 'MPV': MPV_list, "Integ_time": min_max_list, "avg": avg_list, "MPV_error": error_list, "AVG_error": error_avg_list, "Tot_cluster": tot_list}
        result_pd = pd.DataFrame(data=d)

        c = R.TCanvas()
        root_hist_list[0].Draw("histo")
        for h in root_hist_list[1:]:
            h.Draw("histo same")
        c.Draw()
        # c.Print(f"{self.out_folder}/fit_root_1d_pl{planar}_view{view}.pdf")
        return result_pd

    def analize_all_planar_2D(self, macrorun=0):

        nbins = 1000
        range_b = 300
        root_hist_list = []

        for cluster_pd, run in zip(self.cluster_pd_2D_list, self.run_list):
            if macrorun == 0:
                df_c = cluster_pd[(cluster_pd.planar == 1) | (cluster_pd.planar == 2) ]
            else:
                df_c=cluster_pd
            h_n = self.fill_hist_and_norm_1(df_c, nbins=nbins, range_b=range_b)
            root_hist_list.append(h_n)
        fit_list = []
        error_list = []
        for n, histo in enumerate(root_hist_list):
            histo.SetLineColor(n + 1)
            histo.Fit("landau", "ww S")
            histo.GetFunction("landau").SetLineColor(n + 1)
            fit_list.append(histo.GetFunction("landau").GetParameters())
            error_list.append(histo.GetFunction("landau").GetParError(1))
        lines_list = []
        for fit, run in zip(fit_list, self.run_list):
            y_data = self.calculate_y_landau(fit, nbins=nbins, range_b=range_b)
            lines_list.append(go.Scatter(x=np.arange(0, range_b, range_b / nbins), y=y_data, name=f"Fit run {run}"))
        hist_list = []
        for cluster_pd, run in zip(self.cluster_pd_2D_list, self.run_list):
            df_c =  cluster_pd[(cluster_pd.planar == 1) | (cluster_pd.planar == 2) ]

            hist_list.append(go.Histogram(x=df_c.cl_charge,
                                          histnorm="probability",
                                          name=f"Run {run}", opacity=0.75,
                                          xbins=dict(size=range_b / nbins, start=0)))
        fig = go.Figure(hist_list + lines_list)
        fig.update_layout(title="Cluster charge 2D",
                          xaxis_title="Cluster charge [fC]",
                          bargroupgap=0,
                          barmode='overlay',

                          yaxis_title="#")
        fig.show()
        avg_list = []
        error_avg_list = []
        tot_list = []
        for cluster_pd, run in zip(self.cluster_pd_2D_list, self.run_list):
            df_c =  cluster_pd[(cluster_pd.planar == 1) | (cluster_pd.planar == 2) ]
            avg_list.append(np.average(df_c.cl_charge))
            error_avg_list.append(
                np.std(df_c.cl_charge) /
                (df_c.cl_charge.count()) ** (1 / 2)
            )
            tot_list.append(df_c.cl_charge.count())
        min_max_list = []
        MPV_list = []
        for run, FIT in zip(self.run_list, fit_list):
            MPV_list.append(FIT[1])
            min_max_list.append(self.get_min_max_integ(run))
            # print(f"Avg charge cl 1-D {view} run {run}, integ time {self.get_min_max_integ(run)}: {FIT[1]}")
        d = {'run': list(map(str, self.run_list)), 'MPV': MPV_list, "Integ_time": min_max_list, "avg": avg_list, "MPV_error": error_list, "AVG_error": error_avg_list, "Tot_cluster": tot_list}
        result_pd = pd.DataFrame(data=d)

        c = R.TCanvas()
        root_hist_list[0].Draw("histo")
        for h in root_hist_list[1:]:
            h.Draw("histo same")
        c.Draw()
        # c.Print(f"{self.out_folder}/fit_root_1d_pl{planar}_view{view}.pdf")
        return result_pd




    def disply_result_view(self,result_pd, pl, view = "no"):
        if view != "no":
            string=f"view {view}"
        else:
            string=""
        f1=px.scatter(result_pd, x="Integ_time", y="MPV",error_y="MPV_error", color="run", title = f"Planar {pl} Landau MPV cluster {string}")
        f2=px.scatter(result_pd, x="Integ_time", y="avg", error_y="AVG_error",color="run",title = f"Planar {pl} avg cluster {string}" )
        integ_list=result_pd.Integ_time.unique()
        integ_list.sort()
        for integ_time in integ_list:
            result_pd_c=result_pd[result_pd.Integ_time==integ_time]
            print (f"Planar {pl}, view{view} Integ time {integ_time} - MVP max/min {result_pd_c.MPV.min()/result_pd.MPV.max()}")
            print (f"Planar {pl}, view{view} Integ time {integ_time} - avg max/min {result_pd_c.avg.min()/result_pd.avg.max()}")
            print ("----")
        return f1,f2






    def disply_result_2D(self,result_pd, pl, view=False):
        string=""
        if view:
            string=f"View {view}"
        f1=px.scatter(result_pd, x="Integ_time", y="MPV",error_y="MPV_error", color="run", title = f"Planar {pl} Landau MPV cluster 2-D {string}")
        f2=px.scatter(result_pd, x="Integ_time", y="avg", error_y="AVG_error",color="run",title = f"Planar {pl} avg cluster 2-D {string}" )
        integ_list=result_pd.Integ_time.unique()
        integ_list.sort()
        for integ_time in integ_list:
            result_pd_c=result_pd[result_pd.Integ_time==integ_time]
            print (f"Planar {pl}, 2-D Integ time {integ_time} - MVP max/min {result_pd_c.MPV.min()/result_pd.MPV.max()}")
            print (f"Planar {pl}, 2-D Integ time {integ_time} - avg max/min {result_pd_c.avg.min()/result_pd.avg.max()}")
            print ("----")
        return f1,f2

