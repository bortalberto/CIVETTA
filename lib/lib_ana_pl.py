import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import os

standard_template = go.layout.Template()
width_d=1000
height_d=600
standard_template.layout =     dict(
    autosize=False,
    width=width_d,
    height=height_d,
    font_size=18,
    xaxis=dict(
    showline=False,
    zeroline=False,
    gridwidth=1,
    zerolinewidth=2,
    gridcolor="grey",
    zerolinecolor="black",
    linewidth=2, linecolor='black'),
    yaxis=dict(
    showline=False,
    zeroline=False,
    gridwidth=1,
    zerolinewidth=2,
    gridcolor="grey",
    zerolinecolor="black",
    linewidth=2, linecolor='black'),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1

    )
)

raw_root_folder="/home/ihep_data/CIVETTA/planar_data/raw_root/"


def distr_charge_plot_1D_kde(cl_pd, view, k="auto"):
    cl_pd=cl_pd[cl_pd[f"cl_pos_{view}"].notnull()]
    cl_pd_g=cl_pd.groupby("run")
    fig=go.Figure()
    data_dict={}
    for run in cl_pd_g.groups:
        sel_c=cl_pd_g.get_group(run)
        sel_c=sel_c[(sel_c.cl_charge<300) & (sel_c.cl_charge>0)]
        data_dict[run]= np.histogram(sel_c.cl_charge, bins=300, range=[0,300], density=True)
        fig.add_traces(go.Scatter(x=data_dict[run][1], y = data_dict[run][0],
                          line=dict(width = 2, shape='hvh'),name=run))

        if k=="auto":
            sigma=np.std(sel_c.cl_charge)
            q75, q25 = np.percentile(sel_c.cl_charge, [75 ,25])
            iqr = q75 - q25
            k=0.9*np.min([sigma, iqr/1.34])*len(sel_c)**(-1/5)

        kde=KernelDensity(kernel='gaussian', bandwidth=k).fit(sel_c.cl_charge.to_numpy().reshape(-1,1))
        Y=np.exp(kde.score_samples(np.linspace(0, 300, 2000).reshape(-1, 1)))
        X=np.linspace(0, 300, 2000)
        fig.add_trace(go.Scatter(x=X,y=(Y), mode="lines", line=dict(color='royalblue', width=1, dash='dot')))
        print(X[np.argmax(Y)])


    return fig

def extract_kde_and_avg(run, view, planar, k="auto"):
    sel_c=get_run_data([run],"s")
    sel_c=sel_c[(sel_c[f"cl_pos_{view}"].notnull()) & (sel_c.planar==planar)]
    sel_c=sel_c[(sel_c.cl_charge<300) & (sel_c.cl_charge>0)]
    avg=sel_c.cl_charge.mean()
    std=sel_c.cl_charge.std()
    error_avg=std/(len(sel_c)**(1/2))
    sel_c=sel_c[(sel_c.cl_charge>7)]
#     if k=="auto":
#         sigma=np.std(sel_c.cl_charge)
#         q75, q25 = np.percentile(sel_c.cl_charge, [75 ,25])
#         iqr = q75 - q25
#         k=0.9*np.min([sigma, iqr/1.34])*len(sel_c)**(-1/5)
#         print (k)
#     kde=KernelDensity(kernel='gaussian', bandwidth=k).fit(sel_c.cl_charge.to_numpy().reshape(-1,1))
#     Y=np.exp(kde.score_samples(np.linspace(0, 150, 2500).reshape(-1, 1)))
#     X=np.linspace(0, 150, 2500)

#     return X[np.argmax(Y)], avg, error_avg

    return 0, avg, error_avg

def get_run_data(runs, dtype="h"):
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
        data_list.append(pd.read_pickle(f"{raw_root_folder}/{run}/{filename}.pickle.gzip", compression="gzip"))

    return pd.concat(data_list)

def write_html(fig, name, width=width_d, height=height_d):
    """
    Write HTML files for export
    fig, name
    """
    fig.update_layout(
        width=width,
        height=height
    )
    fig.write_html(f"/home/ihep_data/CIVETTA/data/raw_root/export_pdf/{name}.html", include_plotlyjs="directory")



def write_pdf(fig, name, width=width_d, height=height_d):
    """
    Write HTML files for export
    fig, name
    """
#     fig.write_html(f"html_export/{name}.html", include_plotlyjs="CDN")
    fig.write_image(f"/home/ihep_data/CIVETTA/data/raw_root/export_pdf/{name}.pdf",width=width, height=height, scale=1)

colorscale=[[0.0, "rgba(255,255,255,0.0)"],
                [0.0000000000011111, "#0d0887"],
                [0.1111111111111111, "#46039f"],
                [0.2222222222222222, "#7201a8"],
                [0.3333333333333333, "#9c179e"],
                [0.4444444444444444, "#bd3786"],
                [0.5555555555555556, "#d8576b"],
                [0.6666666666666666, "#ed7953"],
                [0.7777777777777778, "#fb9f3a"],
                [0.8888888888888888, "#fdca26"],
                [1.0, "#f9d221"]]
def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]
def charge_vs_time_plot_log(data_pd_cut_2, view="e", log=True):
    if view!="e":
        data_pd_cut_2=data_pd_cut_2[data_pd_cut_2[f"strip_{view}"]>=0]
    hist_2d=np.histogram2d(data_pd_cut_2.charge_SH,-data_pd_cut_2.l1ts_min_tcoarse*6.25*10**-3, bins=[200,267], range=[[0,60],[-1567*6.25*10**-3,-1300*6.25*10**-3]])
    x_bin=((hist_2d[1][1:] + hist_2d[1][:-1])/2)
    y_bin=((hist_2d[2][1:] + hist_2d[2][:-1])/2)
    if log:
        z_value=np.log10(hist_2d[0])
    else:
        z_value=hist_2d[0]
    fig = go.Figure(go.Heatmap(x=y_bin,y=x_bin,z=z_value, colorscale="viridis", coloraxis="coloraxis"))
    fig.update_layout(template=standard_template)

    if log:
        z_max=np.round(np.max(z_value))
        z_max2=np.round(z_max/2)
        fig.update_layout(
            coloraxis_colorbar=dict(
            exponentformat="e",
            title="Hits #",
            tickvals=[0,np.round(z_max2), z_max],
            ticktext=[0,f"{10**np.round(z_max2):.0e}", f"{10**z_max:.0e}" ]))
    fig.update_coloraxes(colorscale="viridis", cmax=z_max, cmin=0)
    fig.update_layout(
        xaxis_title="Time w.r.t. trigger [µs]",
        yaxis_title="Charge [fC]",
        )

    fig.update_yaxes( showgrid=False)
    fig.update_xaxes( showgrid=False)


    return fig

def charge_vs_strip_log(data_pd_cut_2, view="e", log=True):
    if view!="e":
        data_pd_cut_2=data_pd_cut_2[data_pd_cut_2[f"strip_{view}"]>=0]
    hist_2d=np.histogram2d(data_pd_cut_2.charge_SH,data_pd_cut_2[f"strip_{view}"], bins=[200,128], range=[[0,60],[0,128]])
    x_bin=((hist_2d[1][1:] + hist_2d[1][:-1])/2)
    y_bin=((hist_2d[2][1:] + hist_2d[2][:-1])/2)
    if log:
        z_value=np.log10(hist_2d[0])
    else:
        z_value=hist_2d[0]
    fig = go.Figure(go.Heatmap(x=y_bin,y=x_bin,z=z_value, colorscale="viridis", coloraxis="coloraxis"))
    fig.update_layout(template=standard_template)

    if log:
        z_max=np.round(np.max(z_value))
        z_max2=np.round(z_max/2)
        fig.update_layout(
            coloraxis_colorbar=dict(
            exponentformat="e",
            title="Hits #",
            tickvals=[0,np.round(z_max2), z_max],
            ticktext=[0,f"{10**np.round(z_max2):.0e}", f"{10**z_max:.0e}" ]))
    fig.update_coloraxes(colorscale="viridis", cmax=z_max, cmin=0)
    fig.update_layout(
        xaxis_title=f"Strip {view}",
        yaxis_title="Charge [fC]",
        )

    fig.update_yaxes( showgrid=False)
    fig.update_xaxes( showgrid=False)


    return fig
def charge_vs_time_plot_log_multi(data_pd_cut_2, view="e", log=True, titles=["False"]):
    if view!="e":
            data_pd_cut_2=data_pd_cut_2[data_pd_cut_2[f"strip_{view}"]>=0]

    data_gr=data_pd_cut_2.groupby("runNo")
    if titles[0]=="False":
        subplot_titles=([f"Run {run}" for run in list(data_gr.groups)])
    else:
        subplot_titles=titles

    fig = make_subplots(rows=1, cols=data_gr.ngroups, subplot_titles=subplot_titles)
    z_max=0
    for n,run in enumerate(data_gr.groups):
        run_pd=data_gr.get_group(run)
        hist_2d=np.histogram2d(run_pd.charge_SH,-run_pd.l1ts_min_tcoarse*6.25*10**-3, bins=[2,70], range=[[0,60],[-1430*6.25*10**-3,-1361*6.25*10**-3]])
        x_bin=((hist_2d[1][1:] + hist_2d[1][:-1])/2)
        y_bin=((hist_2d[2][1:] + hist_2d[2][:-1])/2)
        if log:
            z_value=np.log10(hist_2d[0])
        else:
            z_value=hist_2d[0]
        z_max=max(np.round(np.max(z_value)), z_max)

        fig.append_trace(go.Heatmap(x=y_bin,y=x_bin,z=z_value, colorscale="viridis", coloraxis="coloraxis", name=run), row=1, col=n+1)
    fig.update_layout(template=standard_template)

    z_max2=np.round(z_max/2)

    fig.update_layout(
        coloraxis_colorbar=dict(
        exponentformat="e",
        title="Hits #",
        tickvals=[0,z_max/2, z_max],
        ticktext=[0,f"{10**np.round(z_max2):.0e}", f"{10**z_max:.0e}" ]))
    fig.update_coloraxes(colorscale="viridis", cmax=z_max, cmin=0)
    fig.update_layout(
        xaxis_title="Time w.r.t. trigger [µs]",
        yaxis_title="Charge [fC]",
        )
    fig.update_yaxes( showgrid=False)
    fig.update_xaxes( showgrid=False)

    return fig


def charge_vs_time_plot_log(data_pd_cut_2, view="e", log=True):
    if view!="e":
        data_pd_cut_2=data_pd_cut_2[data_pd_cut_2[f"strip_{view}"]>0]
    hist_2d=np.histogram2d(data_pd_cut_2.charge_SH,-data_pd_cut_2.l1ts_min_tcoarse*6.25*10**-3, bins=[200,267], range=[[0,60],[-1567*6.25*10**-3,-1300*6.25*10**-3]])
    x_bin=((hist_2d[1][1:] + hist_2d[1][:-1])/2)
    y_bin=((hist_2d[2][1:] + hist_2d[2][:-1])/2)
    if log:
        z_value=np.log10(hist_2d[0])
    else:
        z_value=hist_2d[0]
    fig = go.Figure(go.Heatmap(x=y_bin,y=x_bin,z=z_value, colorscale="viridis", coloraxis="coloraxis"))
    fig.update_layout(template=standard_template)

    if log:
        z_max=np.round(np.max(z_value))
        z_max2=np.round(z_max/2)
        fig.update_layout(
            coloraxis_colorbar=dict(
            exponentformat="e",
            title="Hits #",
            tickvals=[0,np.round(z_max2), z_max],
            ticktext=[0,f"{10**np.round(z_max2):.0e}", f"{10**z_max:.0e}" ]))
    fig.update_coloraxes(colorscale="viridis", cmax=z_max, cmin=0)
    fig.update_layout(
        xaxis_title="Time w.r.t. trigger [µs]",
        yaxis_title="Charge [fC]",
        )

    fig.update_yaxes( showgrid=False)
    fig.update_xaxes( showgrid=False)


    return fig

def charge_vs_time_plot_log_multi(data_pd_cut_2, view="e", log=True, titles=["False"], run_list=["False"]):
    if view!="e":
            data_pd_cut_2=data_pd_cut_2[data_pd_cut_2[f"strip_{view}"]>0]

    data_gr=data_pd_cut_2.groupby("runNo")
    if titles[0]=="False":
        subplot_titles=([f"Run {run}" for run in list(data_gr.groups)])
    else:
        subplot_titles=titles

    fig = make_subplots(rows=1, cols=data_gr.ngroups, subplot_titles=subplot_titles, shared_xaxes=True)
    z_max=0
    if run_list[0]=="False":
        run_list=data_gr.groups
    for n,run in enumerate(run_list):
        run_pd=data_gr.get_group(run)
        hist_2d=np.histogram2d(run_pd.charge_SH,-run_pd.l1ts_min_tcoarse*6.25*10**-3, bins=[800,70], range=[[0,60],[-1430*6.25*10**-3,-1361*6.25*10**-3]])
        x_bin=((hist_2d[1][1:] + hist_2d[1][:-1])/2)
        y_bin=((hist_2d[2][1:] + hist_2d[2][:-1])/2)
        if log:
            z_value=np.log10(hist_2d[0])
        else:
            z_value=hist_2d[0]
        z_max=max(np.round(np.max(z_value)), z_max)

        fig.append_trace(go.Heatmap(x=y_bin,y=x_bin,z=z_value, colorscale="viridis", coloraxis="coloraxis", name=run), row=1, col=n+1)
    fig.update_layout(template=standard_template)

    z_max2=np.round(z_max/2)

    fig.update_layout(
        coloraxis_colorbar=dict(
        exponentformat="e",
        title="Hits #",
        tickvals=[0,z_max/2, z_max],
        ticktext=[0,f"{10**np.round(z_max2):.0e}", f"{10**z_max:.0e}" ]))
    fig.update_coloraxes(colorscale="viridis", cmax=z_max, cmin=0)
    fig.update_layout(
        xaxis_title="Time w.r.t. trigger [µs]",
        yaxis_title="Charge [fC]",
        )
    fig.update_yaxes( showgrid=False)
    fig.update_xaxes( showgrid=False)

    return fig

def charge_vs_time_plot_log_multi_angle(data_pd_cut_2, view="e", log=True, titles=["False"],runs_order=["False"]):
    if view!="e":
            data_pd_cut_2=data_pd_cut_2[data_pd_cut_2[f"strip_{view}"]>0]

    data_gr=data_pd_cut_2.groupby("runNo")
    if titles[0]=="False":
        subplot_titles=([f"Run {run}" for run in list(data_gr.groups)])
    else:
        subplot_titles=titles

    fig = make_subplots(rows=2, cols=data_gr.ngroups, subplot_titles=subplot_titles, vertical_spacing=0.2)
    z_max=0
    if runs_order==["False"]:
        runs_order=data_gr.groups

    for n,run in enumerate(runs_order):
        run_pd=data_gr.get_group(run)
        run_pd=run_pd[run_pd["strip_x"]>0]
        hist_2d=np.histogram2d(run_pd.charge_SH,-run_pd.l1ts_min_tcoarse*6.25*10**-3, bins=[200,70], range=[[0,60],[-1430*6.25*10**-3,-1361*6.25*10**-3]])
        x_bin=((hist_2d[1][1:] + hist_2d[1][:-1])/2)
        y_bin=((hist_2d[2][1:] + hist_2d[2][:-1])/2)
        if log:
            z_value=np.log10(hist_2d[0])
        else:
            z_value=hist_2d[0]
        z_max=max(np.round(np.max(z_value)), z_max)

        fig.append_trace(go.Heatmap(x=y_bin,y=x_bin,z=z_value, colorscale="viridis", coloraxis="coloraxis", name=run), row=1, col=n+1)
        run_pd=data_gr.get_group(run)
        run_pd=run_pd[run_pd["strip_y"]>0]
        hist_2d=np.histogram2d(run_pd.charge_SH,-run_pd.l1ts_min_tcoarse*6.25*10**-3, bins=[200,70], range=[[0,60],[-1430*6.25*10**-3,-1361*6.25*10**-3]])
        x_bin=((hist_2d[1][1:] + hist_2d[1][:-1])/2)
        y_bin=((hist_2d[2][1:] + hist_2d[2][:-1])/2)
        if log:
            z_value=np.log10(hist_2d[0])
        else:
            z_value=hist_2d[0]
        z_max=max(np.round(np.max(z_value)), z_max)

        fig.append_trace(go.Heatmap(x=y_bin,y=x_bin,z=z_value, colorscale="viridis", coloraxis="coloraxis", name=run), row=2, col=n+1)

    fig.update_layout(template=standard_template)

    z_max2=np.round(z_max/2)

    fig.update_layout(
        coloraxis_colorbar=dict(
        title="Hits #",
        thicknessmode="pixels", thickness=30,
        lenmode="pixels", len=400,
        yanchor="top", y=1,
        tickvals=[0,z_max/2, z_max],
        ticktext=[0,f"{10**np.round(z_max2):.0e}", f"{10**z_max:.0e}" ]))
    fig.update_coloraxes(colorscale="viridis", cmax=z_max, cmin=0)
    fig.update_layout(
        xaxis_title="Time w.r.t. trigger [µs]",
        yaxis_title="Charge [fC]",
        )
    fig.update_layout(showlegend=False)
    fig.update_layout(width=1000, height=1000)
    fig.update_yaxes( showgrid=False)
    fig.update_xaxes( showgrid=False)
    fig.update_yaxes( title="Charge [fC]", row=2, col=1)
    fig.update_xaxes( title="Time w.r.t. trigger [µs]", row=2, col=1)
    return fig


def charge_vs_time_plot_log_multi_drift(data_pd_cut_2, view="e", log=True, titles=["False"]):
    if view!="e":
            data_pd_cut_2=data_pd_cut_2[data_pd_cut_2[f"strip_{view}"]>0]

    data_gr=data_pd_cut_2.groupby("runNo")
    if titles[0]=="False":
        subplot_titles=([f"Run {run}" for run in list(data_gr.groups)])
    else:
        subplot_titles=titles


    fig = make_subplots(rows=2, cols=data_gr.ngroups, subplot_titles=subplot_titles+subplot_titles,vertical_spacing=0.2, row_heights=[0.7,0.3])
    print (len (fig.layout.annotations))
    z_max=0
    annotations=list(fig.layout["annotations"])
    for n,run in enumerate(data_gr.groups):
        run_pd=data_gr.get_group(run)
        hist_2d=np.histogram2d(run_pd.charge_SH,-run_pd.l1ts_min_tcoarse*6.25*10**-3, bins=[200,70], range=[[0,60],[-1430*6.25*10**-3,-1361*6.25*10**-3]])
        x_bin=((hist_2d[1][1:] + hist_2d[1][:-1])/2)
        y_bin=((hist_2d[2][1:] + hist_2d[2][:-1])/2)
        if log:
            z_value=np.log10(hist_2d[0])
        else:
            z_value=hist_2d[0]
        z_max=max(np.round(np.max(z_value)), z_max)

        fig.append_trace(go.Heatmap(x=y_bin,y=x_bin,z=z_value, colorscale="viridis", coloraxis="coloraxis", name=run), row=1, col=n+1)
        run_pd_c=run_pd[(run_pd.charge_SH>15)]
        hist_1d=np.histogram(-run_pd_c.l1ts_min_tcoarse*6.25*10**-3, bins=70, density=True, range=[-1430*6.25*10**-3,-1361*6.25*10**-3])
        hmx = half_max_x(hist_1d[1],hist_1d[0])
        fwhm = hmx[1] - hmx[0]
        print("FWHM:{:.3f}".format(fwhm))
        fig.append_trace(go.Scatter(x=hist_1d[1], y = hist_1d[0]/100,
                          line=dict(width = 2, shape='hvh', color="black")), row=2, col=n+1)
#         annotations.append(
#                 dict(
#                     x=-8.4, y=0.16, # annotation point
#                     xref=f'x{data_gr.ngroups+n+1}',
#                     yref=f'y{data_gr.ngroups+n+1}',
#                     text="FWHM:{:.3f}".format(fwhm),
#                     showarrow=False
#                     ))
        fig.layout.annotations[data_gr.ngroups+n].update(text="FWHM:{:.3f}µs".format(fwhm))
#     fig.update_layout(annotations=annotations)
    fig.update_layout(template=standard_template)
    z_max2=np.round(z_max/2)
    fig.update_layout(
        coloraxis_colorbar=dict(
        title="Hits #",
        thicknessmode="pixels", thickness=30,
        lenmode="pixels", len=400,
        yanchor="top", y=1,
        tickvals=[0,z_max/2, z_max],
        ticktext=[0,f"{10**np.round(z_max2):.0e}", f"{10**z_max:.0e}" ]))
    fig.update_coloraxes(colorscale="viridis", cmax=z_max, cmin=0)
    fig.update_layout(
        xaxis_title="Time w.r.t. trigger [µs]",
        yaxis_title="Charge [fC]",
        )
    fig.update_layout(showlegend=False)
    fig.update_layout(width=1000, height=1000)
    fig.update_yaxes( showgrid=False)
    fig.update_xaxes( showgrid=False)
    fig.update_yaxes( title="Frequency", row=2, col=1)
    fig.update_xaxes( title="Time w.r.t. trigger [µs]", row=2, col=1)

    return fig

def charge_vs_time_plot(data_pd_cut_2, view="e"):
    if view!="e":
        data_pd_cut_2=data_pd_cut_2[data_pd_cut_2[f"strip_{view}"]>0]
    fig = px.density_heatmap(data_pd_cut_2, x="l1ts_min_tcoarse", y="charge_SH",
                             title="Charge vs time",
                             marginal_x="histogram",
                             marginal_y="histogram",
                             nbinsx=int(data_pd_cut_2.l1ts_min_tcoarse.max() - (data_pd_cut_2.l1ts_min_tcoarse.min())),
                             nbinsy=120)
    fig.update_layout(
        xaxis_title="Trigger time stamp - hit TCoarse ",
        yaxis_title="Charge [fC]",
    )
    fig.update_xaxes(range=[1300, 1600])
    fig.update_yaxes(range=[0, 60])
    return fig

def strips_plot(data_pd_cut_2, view):
    data_pd_cut_3 = data_pd_cut_2[data_pd_cut_2[f"strip_{view}"] > 0]
    fig = px.density_heatmap(data_pd_cut_3, x=f"strip_{view}", y="charge_SH",
                             title=f"Charge vs strip {view}",
                             marginal_x="histogram",
                             color_continuous_scale=colorscale,
                             hover_data={'channel': True,  # remove species from hover data
                                         f'strip_{view}': True,  # customize hover for column of y attribute
                                         'gemroc' : True  # add other column, default formatting
                                         },
                             nbinsx=128,
                             range_x=[0, 128],
                             nbinsy=120,
                             range_y=[0, 60])
    fig.update_layout(
        yaxis_title="Charge [fC] ",
        xaxis_title=f"{view} strip",
        height=800
    )
    return fig

def signal_ratio_plot(data_pd_cut_2, sel_run, planar):
    ## Estract time informmation
    time = {}
    time_reader = log_loader_time.reader(data_folder + "/raw_dat/", data_folder + "/time/")
    time_dict=time_reader.elab_on_run_dict(sel_run)
    start, end = time_dict
    time[int(sel_run)] = calculater_middle_time(start, end)
    ## Add time information to PD
    data_pd_cut_2.insert(0, "time_r", 0)
    data_pd_cut_2["time_r"] = data_pd_cut_2.apply(lambda row: time[(int(row.runNo))][str(int(row.subRunNo))], axis=1)
    data_pd_cut_2['time_r'] = pd.to_datetime(data_pd_cut_2['time_r'], unit='s')
    signal_list = []

    ## Create signal and noise count PD
    sin_cut = data_pd_cut_2[((data_pd_cut_2.l1ts_min_tcoarse < signal_upper_limit) & (data_pd_cut_2.l1ts_min_tcoarse > signal_lower_limit) & (data_pd_cut_2.planar == planar) & (data_pd_cut_2.charge_SH > 5))]
    nois_cut = data_pd_cut_2[((data_pd_cut_2.l1ts_min_tcoarse > signal_upper_limit) | (data_pd_cut_2.l1ts_min_tcoarse < signal_lower_limit) & (data_pd_cut_2.planar == planar))]

    sin = sin_cut.groupby(sin_cut.time_r.dt.hour).agg({"charge_SH": "count", "time_r": "last"})
    nois = nois_cut.groupby(nois_cut.time_r.dt.hour).agg({"charge_SH": "count", "time_r": "last"})

    signal_list.append(nois.rename(columns={"charge_SH": "noise_planar_{}".format(planar)}))
    signal_list.append(sin.rename(columns={"charge_SH": "signal_planar_{}".format(planar)}))
    result = pd.concat(signal_list, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    x_data=[x for x in result.time_r]
    y_data=[y for y in result[f"signal_planar_{planar}"]/result[f"noise_planar_{planar}"]]
    fig = px.scatter(
            x=x_data,
            y=y_data
        )

    fig.update_layout(
        yaxis_title="Signal/noise  ratio",
        xaxis_title="Time",
        height=800
    )
    if max(y_data)<1:
        fig.update_yaxes(range=[0, 1])
    return fig


def noise_hits_plot(data_pd_cut_2, sel_run, planar):
    ## Estract time informmation
    time = {}
    time_reader = log_loader_time.reader(data_folder + "/raw_dat/", data_folder + "/time/")
    time_dict = time_reader.elab_on_run_dict(sel_run)
    start, end = time_dict
    time[int(sel_run)] = calculater_middle_time(start, end)
    ## Add time information to PD
    data_pd_cut_2.insert(0, "time_r", 0)
    data_pd_cut_2["time_r"] = data_pd_cut_2.apply(lambda row: time[(int(row.runNo))][str(int(row.subRunNo))], axis=1)
    data_pd_cut_2['time_r'] = pd.to_datetime(data_pd_cut_2['time_r'], unit='s')

    ## Create noise count PD
    nois_cut = data_pd_cut_2[((data_pd_cut_2.l1ts_min_tcoarse > signal_upper_limit) | (data_pd_cut_2.l1ts_min_tcoarse < signal_lower_limit) & (data_pd_cut_2.planar == planar))]
    noise_pd = nois_cut.groupby(nois_cut.subRunNo).agg({"charge_SH": "count", "time_r": "last", "subRunNo": "last"}, )
    noise_pd = noise_pd.rename(columns={"charge_SH": "noise_tot"})
    trig_dict = extract_num_triggers_run(data_folder + "/raw_dat/", sel_run)
    noise_win_width = (1567 - signal_upper_limit + signal_lower_limit - 1299) * 6.25 * 10 ** (-9)
    noise_pd["noise_rate"] = noise_pd.apply(calc_rate_per_sub, args=(trig_dict, noise_win_width), axis=1)
    # noise_pd = noise_pd.groupby(nois_cut.time_r.dt.hour).agg({"noise_rate": "mean", "time_r": "last","subRunNo":"last"}, )
    noise_pd["time_r"] = (noise_pd["time_r"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    noise_pd = noise_pd.groupby(noise_pd.subRunNo.divide(5).round().multiply(5)).agg({"subRunNo": "mean", "time_r": "mean", "noise_rate": "mean"})
    noise_pd["time_r"] = pd.to_datetime(noise_pd["time_r"], unit='s')
    fig = px.scatter(
        noise_pd,
        x="time_r",
        y="noise_rate",
        hover_data=["subRunNo"]
    )

    fig.update_layout(
        yaxis_title="Noise planar [Hz]",
        xaxis_title="Time",
        height=800
    )

    if max(noise_pd.noise_rate) < 5000000:
        fig.update_yaxes(range=[0, 5000000])
    return fig

## Cluster plots

def signal_heatmap_plot(cluster_pd_2D_pre_2, sel_binning):
    fig = px.density_heatmap(x=cluster_pd_2D_pre_2.cl_pos_x*0.0650, y=cluster_pd_2D_pre_2.cl_pos_y*0.0650, marginal_x="histogram", marginal_y="histogram", nbinsx=int(128 / sel_binning), nbinsy=int(128 / sel_binning), color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(
        yaxis_title="Y pos [cm] ",
        xaxis_title="X pos [cm]",
        height=800,
        width=800
    )
    return fig

def clusters_vs_tim_plot(cluster_pd_2D_pre_2, sel_run):
    time = {}
    time_reader = log_loader_time.reader(data_folder + "/raw_dat/", data_folder + "/time/")
    time_dict = time_reader.elab_on_run_dict(sel_run)
    start, end = time_dict
    time[int(sel_run)] = calculater_middle_time(start, end)
    ## Add time information to PD
    cluster_pd_2D_pre_2.insert(0, "time_r", 0)
    cluster_pd_2D_pre_2["time_r"] = cluster_pd_2D_pre_2.apply(lambda row: time[(int(row.run))][str(int(row.subrun))], axis=1)
    cluster_pd_2D_pre_2['time_r'] = pd.to_datetime(cluster_pd_2D_pre_2['time_r'], unit='s')
    clusters_list = []
    trigger_list = []
    time_list = []
    trig_dict = extract_num_triggers_run(data_folder + "/raw_dat/", sel_run)
    for subrun in cluster_pd_2D_pre_2.subrun.unique():
        trigger_list.append(trig_dict[str(int(subrun))])
        cl_pd = cluster_pd_2D_pre_2[cluster_pd_2D_pre_2.subrun == subrun]
        clusters_list.append(cl_pd.cl_pos_x.count())
        time_list.append(cl_pd.time_r.min())
    fig = px.scatter(x=time_list, y=np.divide(clusters_list, trigger_list), marginal_y="histogram")
    fig.update_layout(
        yaxis_title="Clusters / trigger",
        xaxis_title="Time",
        height=800
    )
    return fig
def distr_charge_plot_2D(cluster_pd_2D_pre_2):
    range_b = 500
    nbins = 500
    cluster_pd_2D_cut = cluster_pd_2D_pre_2[cluster_pd_2D_pre_2.cl_charge < 500]
    fit = fill_hist_and_norm_and_fit_landau(cluster_pd_2D_cut, 1, nbins, range_b)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=cluster_pd_2D_cut.cl_charge, opacity=0.75, xbins=dict(size=range_b / nbins, start=0), name="Charge histogram"))
    fig.update_layout(
        yaxis_title="Count",
        xaxis_title="Cluster charge [fC]",
        height=800
    )
    x_list = np.arange(0, range_b, range_b / nbins)
    y_list = [calculate_y_landau(fit, x) for x in x_list]
    fig.add_trace(go.Scatter(x=x_list, y=y_list, name="Landau fit"))
    textfont = dict(
        family="sans serif",
        size=18,
        color="LightSeaGreen"
    )
    fig.add_annotation(x=0.75, xref="paper", y=0.95, yref="paper",
                       text=f"""AVG Charge: {cluster_pd_2D_cut.cl_charge.mean():.2f} +/- {
                       np.std(cluster_pd_2D_cut.cl_charge) /
                       (cluster_pd_2D_cut.cl_charge.count()) ** (1 / 2):.2f}""", showarrow=False, font=textfont)
    fig.add_annotation(x=0.75, xref="paper", y=0.90, yref="paper",
                       text=f"MPV CHarge: {fit[1]:.2f} +/- {fit[4]:.2f}", showarrow=False, font=textfont)
    fig.add_annotation(x=0.75, xref="paper", y=0.85, yref="paper",
                       text=f"""AVG size(x+y): {cluster_pd_2D_cut.cl_size_tot.mean():.2f} +/- {
                       np.std(cluster_pd_2D_cut.cl_size_tot) /
                       (cluster_pd_2D_cut.cl_size_tot.count()) ** (1 / 2):.2f}""", showarrow=False, font=textfont)

    return fig
def distr_size_plot_confronto(cl_pd,cl_pd2,cl_pd_s, view):
    cl_pd=cl_pd[cl_pd[f"cl_pos_{view}"].notnull()]
    cl_pd2=cl_pd2[cl_pd2[f"cl_pos_{view}"].notnull()]
    cl_pd_s=cl_pd_s[cl_pd_s[f"cl_pos_{view}"].notnull()]

    data_dict={}
    data_dict["All clusters"]= np.histogram(cl_pd.cl_size, bins=7, range=[0,7], density=True )
    data_dict["2D selection"]= np.histogram(cl_pd2[f"cl_size_{view}"], bins=7, range=[0,7], density=True )
    data_dict["track selection"]= np.histogram(cl_pd_s.cl_size, bins=7, range=[0,7], density=True )

    fig = go.Figure()
    for run in data_dict:
        fig.add_traces(go.Scatter(x=data_dict[run][1], y = data_dict[run][0],
                          line=dict(width = 2, shape='hvh'), mode="lines",name=run))

    return fig

def distr_charge_plot_confronto(cl_pd,cl_pd2,cl_pd_s, view):
    cl_pd=cl_pd[cl_pd[f"cl_pos_{view}"].notnull()]
    cl_pd2=cl_pd2[cl_pd2[f"cl_pos_{view}"].notnull()]
    cl_pd_s=cl_pd_s[cl_pd_s[f"cl_pos_{view}"].notnull()]

    data_dict={}
    data_dict["All clusters"]= np.histogram(cl_pd.cl_charge, bins=300, range=[0,150] )
    data_dict["2D selection"]= np.histogram(cl_pd2[f"cl_charge_{view}"], bins=300, range=[0,150] )
    data_dict["track selection"]= np.histogram(cl_pd_s.cl_charge, bins=300, range=[0,150] )

    fig = go.Figure()
    for run in data_dict:
        fig.add_traces(go.Scatter(x=data_dict[run][1], y = data_dict[run][0],
                          line=dict(width = 2, shape='hvh'),name=run))

    return fig

def distr_res_plot(cl_pd, view,planar):
    cl_pd=cl_pd[cl_pd[f"res_planar_{planar}_{view}"].notnull()]
    cl_pd_g=cl_pd.groupby("run")
    data_dict={}
    for run in cl_pd_g.groups:
        cl_pd=cl_pd_g.get_group(run)
        data_dict[run]= np.histogram(cl_pd[f"res_planar_{planar}_{view}"], bins=600, range=[-1,1], density=True)
    fig = go.Figure()
    for run in data_dict:
        fig.add_traces(go.Scatter(x=data_dict[run][1], y = data_dict[run][0],
                          line=dict(width = 2, shape='hvh'),name=run))

    return fig


def distr_charge_plot_1D(cl_pd, view, run_list=["false"]):
    cl_pd=cl_pd[cl_pd[f"cl_pos_{view}"].notnull()]
    cl_pd_g=cl_pd.groupby("run")
    data_dict={}
    if run_list[0]=="false":
        run_list=cl_pd_g.groups
    for run in run_list:
        cl_pd=cl_pd_g.get_group(run)
        data_dict[run]= np.histogram(cl_pd.cl_charge, bins=300, range=[0,300], density=True)
    fig = go.Figure()
    for run in data_dict:
        fig.add_traces(go.Scatter(x=data_dict[run][1], y = data_dict[run][0],
                          line=dict(width = 2, shape='hvh'),name=run))

    return fig

def joydiv_distr_charge_plot_1D(cl_pd, view, run_list=["false"]):
    cl_pd=cl_pd[cl_pd[f"cl_pos_{view}"].notnull()]
    cl_pd_g=cl_pd.groupby("run")
    data_dict={}
    if run_list[0]=="false":
        run_list=cl_pd_g.groups
    for run in run_list:
        cl_pd=cl_pd_g.get_group(run)
        data_dict[run]= np.histogram(cl_pd.cl_charge, bins=300, range=[0,300], density=True)
    fig = go.Figure()
    for run in data_dict:
        fig.add_traces(go.Scatter3d(x=data_dict[run][1], z = data_dict[run][0], y=[run],mode="lines",
                          name=run))

    return fig

def distr_size_plot_1D(cl_pd, view, run_list=["false"]):
    cl_pd=cl_pd[cl_pd[f"cl_pos_{view}"].notnull()]
    cl_pd_g=cl_pd.groupby("run")
    data_dict={}
    if run_list[0]=="false":
        run_list=cl_pd_g.groups
    for run in run_list:
        cl_pd=cl_pd_g.get_group(run)
        data_dict[run]= np.histogram(cl_pd.cl_size, bins=20, range=[0,20], density=True)

    fig = go.Figure()
    for run in data_dict:
        fig.add_traces(go.Scatter(x=data_dict[run][1], y = data_dict[run][0],
                          line=dict(width = 2, shape='hvh'),name=run))

    return fig

def distr_pos_plot_1D(cl_pd, view):
    cl_pd=cl_pd[cl_pd[f"cl_pos_{view}"].notnull()]
    cl_pd_g=cl_pd.groupby("run")
    data_dict={}
    for run in cl_pd_g.groups:
        cl_pd=cl_pd_g.get_group(run)
        data_dict[run]= np.histogram(cl_pd[f"cl_pos_{view}"]*0.0650, bins=128, range=[-0.001,128*0.0650], density=False)

    fig = go.Figure()
    for run in data_dict:
        fig.add_traces(go.Scatter(x=data_dict[run][1], y = data_dict[run][0],
                          line=dict(width = 2, shape='hvh'),name=run))

    return fig

def calculate_eff_fast(data, res_tol_tr, res_tol_put,put, view):
    mean_dict = {}
    keys = []
    data_tr = data
    for planarc in range(0, 4):
        for view_t in ("x", "y"):
            mean_dict[f"res_planar_{planarc}_{view_t}"] = data[f"res_planar_{planarc}_{view_t}"].mean()
            keys.append(f"res_planar_{planarc}_{view_t}")
    for key in keys:
        data_tr[key] = data_tr[key] - mean_dict[key]

    res_pd_dict = {
        "planar"   : [],
        "view"     : [],
        "eff"      : [],
        "num"      : [],
        "denom"    : [],
        "error_eff": []
    }
    data_c = data_tr[data_tr[f"{view}_fit"].notna()]
    trackers = [f"res_planar_{planar}_{view}" for planar in range(0, 4) if planar != put]
    data_cc = data_c[data_c[trackers].apply(lambda x: all(x < res_tol_tr), 1)]
    data_put = data_cc[data_cc[f"res_planar_{put}_{view}"] < res_tol_put]
    res_pd_dict["planar"].append(put)
    res_pd_dict["view"].append(view)
    eff = len(data_put) / len(data_cc)
    num = len(data_put)
    denom = len(data_cc)
    error_eff = (float((eff * (1 - eff) / denom) ** (1 / 2)))
    res_pd_dict["eff"].append(eff)
    res_pd_dict["num"].append(num)
    res_pd_dict["denom"].append(denom)
    res_pd_dict["error_eff"].append(error_eff)

    eff_pd = pd.DataFrame(res_pd_dict)
    return (eff,denom)

def  create_result_list():
    kde_list=[]
    avg_list=[]
    error_avg_list=[]
    run_list=[]
    view_list=[]
    planar_list=[]

    for path in os.walk("../../planar_data/raw_root/"):
        if len(path[1])>0:
            runs=path[1]
    for view in tqdm(("x","y")):
        for planar in tqdm((1,3),leave=False):
            for run in tqdm (runs, leave=False):
                try:
                    kde,avg, error_avg= extract_kde_and_avg(int(run), view=view, planar=planar, k=5)
                    run_list.append(run)
                    kde_list.append(kde)
                    avg_list.append(avg)
                    error_avg_list.append(error_avg)
                    view_list.append(view)
                    planar_list.append(planar)
                    print (f"run {run} done" )
                except FileNotFoundError:
                    print (f"No file found for {run}")
                except KeyError as e:
                    print (f"{run} : {e}\n View {view} not found")
                except ValueError as e:
                    print (f"{run} : {e}")
                except Exception as e:
                    print (f"{run} : {e}")
    dict_4_pd={
        "run":run_list,
        "mpv":kde_list,
        "avg":avg_list,
        "error_avg": error_avg_list,
        "view": view_list,
        "planar":planar_list
    }
    result_pd=pd.DataFrame(dict_4_pd)
    result_pd.to_pickle("avg_and_kde_2.pkl")

def add_cl_size():
    kde_list=[]
    error_avg_list=[]
    avg_list=[]
    run_list=[]
    view_list=[]
    planar_list=[]
    for path in os.walk("../../planar_data/raw_root/"):
        if len(path[1])>0:
            runs=path[1]
    for view in tqdm(("x","y")):
        for planar in tqdm((1,3),leave=False):
            for run in tqdm (runs, leave=False):
                try:
                    sel_c=get_run_data([run],"s")
                    sel_c=sel_c[(sel_c[f"cl_pos_{view}"].notnull()) & (sel_c.planar==planar)]
                    hist=np.histogram(sel_c.cl_size, bins=20, range=(0,20))
                    kde=hist[1][hist[0].argmax()]
                    avg= sel_c.cl_size.mean()
                    std=sel_c.cl_size.std()
                    error_avg=std/(len(sel_c)**(1/2))
                    run_list.append(run)
                    kde_list.append(kde)
                    avg_list.append(avg)
                    error_avg_list.append(error_avg)

                    view_list.append(view)
                    planar_list.append(planar)
                    print (f"run {run} done" )
                except FileNotFoundError:
                    print (f"No file found for {run}")
                except KeyError as e:
                    print (f"{run} : {e}\n View {view} not found")
                except ValueError as e:
                    print (f"{run} : {e}")
                except Exception as e:
                    print (f"{run} : {e}")
    dict_4_pd={
        "run":run_list,
        "mpv_size":kde_list,
        "avg_size":avg_list,
        "avg_size_error":error_avg_list,
        "view": view_list,
        "planar":planar_list
    }
    result_pd=pd.DataFrame(dict_4_pd)
    result_pd.to_pickle("avg_and_kde_size_2.pkl")