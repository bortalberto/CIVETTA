import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import configparser
import os
import sys

run_number = sys.argv[1]
config = configparser.ConfigParser()
config.read(os.path.join(sys.path[0], "config_cyl.ini"))
signal_window_lower_limit_conf = config["GLOBAL"].getint("signal_window_lower_limit")
signal_window_upper_limit_conf = config["GLOBAL"].getint("signal_window_upper_limit")
signal_win = [signal_window_lower_limit_conf, signal_window_upper_limit_conf ]
noise_win = [signal_window_lower_limit_conf- 20, signal_window_upper_limit_conf + 20]

data_folder = config["GLOBAL"].get("data_folder")
plot_folder = config["GLOBAL"].get("plot_folder")

out_path = os.path.join(plot_folder, f"{run_number}")
if not os.path.isdir(out_path):
    os.mkdir(out_path)
strip_max = {
    "1X": 856,
    "1V": 1173,
    "2X": 1260,
    "2V": 2154,
    "3X": 1663,
    "3V": 2790
}
standard_template = go.layout.Template()
width_d = 800
height_d = 700
standard_template.layout = dict(
    autosize=False,
    width=width_d,
    height=height_d,
    font_size=18,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1)
)

colorscale = [[0.0, "rgba(255,255,255,0.0)"],
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
print (f"Generating plots for run {run_number}")
run_data = pd.read_feather(os.path.join(data_folder, "raw_root",f"{run_number}","hit_data-zstd.feather"))
def colorbar(n):
    return dict(
        tick0 = 0,
        title = "Log color scale",
        tickmode = "array",
        tickvals = np.linspace(0, n, n+1),
        ticktext = 10**np.linspace(0, n, n+1))

def desity_heatmap_log(run_data_view, stringx, stringy, nbinsx, nbinsy, rangex, rangey):
    histo_data = np.histogram2d(
        run_data_view[f"{stringy}"],
        run_data_view[f"{stringx}"],
        bins=[nbinsy, nbinsx],
        range=[rangey, rangex])
    with np.errstate(divide='ignore'):
        z_data = np.log10(histo_data[0])
    z_data[z_data == -np.inf] = 0
    n = int(np.max((z_data)))
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=histo_data[2],
        y=histo_data[1],
        hovertemplate="time: %{x} <br>" + "charge: %{y} <br>" + "count: 10^%{z}",
        colorbar=colorbar(n + 1),
        colorscale=colorscale))
    fig.update(layout_showlegend=False)

    return histo_data, fig


## Noise plots
for layer in (1, 2, 3):
    for view in ("X", "V"):
        ## Noise:
        run_data_view = run_data.query(f"planar == {layer} and view=='{view}'")
        run_data_view = run_data_view.query(f"l1ts_min_tcoarse<{noise_win[0]} or l1ts_min_tcoarse>{noise_win[1]}")
        fig = px.histogram(
            run_data_view,
            "strip",
            log_y=True,
            nbins=strip_max[f"{layer}{view}"],
            range_x=[0, strip_max[f"{layer}{view}"] + 1])

        #         ## Occupancy noise
        #         fig = desity_heatmap_log(
        #             run_data_view,
        #             stringx="strip",
        #             stringy="charge_SH",
        #             nbinsy=10,
        #             nbinsx=strip_max[f"{layer}{view}"],
        #             rangex=[0,strip_max[f"{layer}{view}"]],
        #             rangey=[-5,run_data_view.charge_SH.quantile(0.99)] )[1]
        fig.update_layout(template=standard_template,
                          title=f"Charge on strip {layer}{view} - noise time cut outside {noise_win} ")
        fig.update_layout(xaxis={"title": f"Strip {view}"})
        fig.update_layout(yaxis={"title": "Count"})
        fig.write_html(f"{out_path}/noise_strip_occupancy{layer}{view}.html", include_plotlyjs="../plotly.min.js")


## All plots
run_data_c = run_data.query(
    f"charge_SH>-5 and l1ts_min_tcoarse>1299 and l1ts_min_tcoarse<1567 and delta_coarse<50 and delta_coarse>4"
)
for layer in (1, 2, 3):
    for view in ("X", "V"):
        run_data_view = run_data_c.query(f"planar == {layer} and view=='{view}'")

        ## Time
        fig = px.histogram(run_data_view, "l1ts_min_tcoarse", range_x=[1299, 1567], nbins=268)
        fig.update_layout(template=standard_template, title=f"Hit time L{layer},{view}")
        fig.update_layout(xaxis={"title": "l1ts_min_tcoarse"})
        fig.update_layout(yaxis={"title": "#hits"})
        fig.write_html(f"{out_path}/time_L{layer}{view}.html", include_plotlyjs="../plotly.min.js")

        ## Carge_vs_time
        fig = px.density_heatmap(run_data_view, "l1ts_min_tcoarse", "charge_SH",
                                 range_x=[1299, 1567], nbinsx=268, range_y=[-5, 65], nbinsy=140,
                                 color_continuous_scale=colorscale)
        fig.update_layout(template=standard_template, title=f"Charge vs time L{layer},{view}")
        fig.update_layout(yaxis={"title": "Charge [fC]"})
        fig.update_layout(xaxis={"title": "l1ts_min_tcoarse"})
        fig.write_html(f"{out_path}/chargevtime_L{layer}{view}.html", include_plotlyjs="../plotly.min.js")

        ## Carge_vs_time_log
        fig = desity_heatmap_log(
            run_data_view,
            stringy="charge_SH",
            stringx="l1ts_min_tcoarse",
            nbinsx=268,
            nbinsy=150,
            rangex=[1299, 1567],
            rangey=[-5, 60])[1]
        fig.update_layout(template=standard_template, title=f"Charge vs time L{layer},{view} log")
        fig.update_layout(yaxis={"title": "Charge [fC]"})
        fig.update_layout(xaxis={"title": "l1ts_min_tcoarse"})
        fig.update_traces(showscale=False)
        fig.write_html(f"{out_path}/chargevtime_L{layer}{view}_log.html", include_plotlyjs="../plotly.min.js")

        ## Signal
        fig = px.density_heatmap(
            run_data_view,
            "strip", "charge_SH",
            range_x=[0, strip_max[f"{layer}{view}"] + 1],
            range_y=[-5, 65],
            nbinsy=140,
            nbinsx=strip_max[f"{layer}{view}"],
            color_continuous_scale=colorscale)
        fig.update_layout(template=standard_template,
                          title=f"Charge on strip {layer}{view} ")
        fig.update_layout(xaxis={"title": f"Strip {view}"})
        fig.update_layout(yaxis={"title": "Charge [fC]"})
        fig.write_html(f"{out_path}/chargevstrip{layer}{view}.html", include_plotlyjs="../plotly.min.js")

        fig = px.histogram(
            run_data_view.query(f"{signal_win[0]}<l1ts_min_tcoarse<{signal_win[1]} and charge_SH>10"),
            "strip",
            nbins=strip_max[f"{layer}{view}"],
            range_x=[0, strip_max[f"{layer}{view}"] + 1])

        #         ## Occupancy noise
        #         fig = desity_heatmap_log(
        #             run_data_view,
        #             stringx="strip",
        #             stringy="charge_SH",
        #             nbinsy=10,
        #             nbinsx=strip_max[f"{layer}{view}"],
        #             rangex=[0,strip_max[f"{layer}{view}"]],
        #             rangey=[-5,run_data_view.charge_SH.quantile(0.99)] )[1]
        fig.update_layout(template=standard_template,
                          title=f"Charge on strip {layer}{view} - signal time cut {signal_win} ")
        fig.update_layout(xaxis={"title": f"Strip {view}"})
        fig.update_layout(yaxis={"title": "Count"})
        fig.write_html(f"{out_path}/signal_strip_occupancy{layer}{view}.html", include_plotlyjs="../plotly.min.js")


## Clusters
if os.path.isfile(os.path.join(data_folder, "raw_root",f"{run_number}","cluster_pd_2D-zstd.feather")):
    cluster1d = pd.read_feather(os.path.join(data_folder, "raw_root",f"{run_number}","cluster_pd_1D-zstd.feather"))
    cluster2d = pd.read_feather(os.path.join(data_folder, "raw_root",f"{run_number}","cluster_pd_2D-zstd.feather"))
    cluster1d["view"] = ""
    cluster1d["cl_pos"] = np.nan
    cluster1d.loc[cluster1d.cl_pos_x > -1, "view"] = "X"
    cluster1d.loc[cluster1d.cl_pos_y > -1, "view"] = "V"
    cluster1d.loc[cluster1d.view == "X", "strip"] = cluster1d.loc[cluster1d.view == "X", "cl_pos_x"]
    cluster1d.loc[cluster1d.view == "V", "strip"] = cluster1d.loc[cluster1d.view == "V", "cl_pos_y"]

    #cluster1d
    for layer in (1, 2, 3):
        for view in ("X", "V"):
            cluster1d_view = cluster1d.query(f"planar == {layer} and view=='{view}'")

            ## Charge
            fig = px.histogram(cluster1d_view, "cl_charge", )
            fig.update_layout(template=standard_template,
                              title=f"Cluster charge {layer}{view} ")
            fig.update_traces(xbins=dict(  # bins used for histogram
                start=0.0,
                end=250.0,
                size=5
            ))
            fig.update_layout(yaxis={"title": "#Clusters"})
            fig.update_layout(xaxis={"title": "Cluster charge [fC]"})

            fig.write_html(f"{out_path}/cluster_charge{layer}{view}.html", include_plotlyjs="../plotly.min.js")


            ## size
            fig = px.histogram(cluster1d_view, "cl_size", )
            fig.update_layout(template=standard_template,
                              title=f"Cluster size {layer}{view} ")
            fig.update_traces(xbins=dict(  # bins used for histogram
                start=0.0,
                end=20.0,
                size=1
            ))
            fig.update_layout(yaxis={"title": "#Clusters"})
            fig.update_layout(xaxis={"title": "Cluster size [strips]"})
            fig.write_html(f"{out_path}/cluster_size{layer}{view}.html", include_plotlyjs="../plotly.min.js")


    #cluster 2D
    for layer in (1, 2, 3):

        ##Charge
        cluster2d_view = cluster2d.query(f"planar == {layer}")

        fig = px.histogram(cluster2d_view, f"cl_charge", )
        fig.update_layout(template=standard_template,
                          title=f"Cluster charge (cluster2d) L{layer} ")
        fig.update_traces(xbins=dict(  # bins used for histogram
            start=0.0,
            end=250.0,
            size=5
        ))
        fig.update_layout(yaxis={"title": "#clusters"})
        fig.update_layout(xaxis={"title": "Cluster charge [fC]"})

        fig.write_html(f"{out_path}/cluster2D_charge{layer}.html", include_plotlyjs="../plotly.min.js")


        ## Size
        fig = px.histogram(cluster2d_view, f"cl_charge", )
        fig.update_layout(template=standard_template,
                          title=f"Cluster charge (cluster2d) L{layer} ")
        fig.update_traces(xbins=dict(  # bins used for histogram
            start=0.0,
            end=250.0,
            size=5
        ))
        fig.update_layout(yaxis={"title": "#hits"})
        fig.update_layout(xaxis={"title": "Cluster size [fC]"})

        fig.write_html(f"{out_path}/cluster2D_size{layer}.html", include_plotlyjs="../plotly.min.js")


        ## Size
        fig = px.density_heatmap(cluster2d_view, x=f"cl_pos_x", y="cl_pos_y")
        fig.update_layout(
            template=standard_template,
            title=f"Cluster position (cluster2d) L{layer} ")
        fig.update_traces(xbins=dict(  # bins used for histogram
            start=0.0,
            end=strip_max[f"{layer}X"],
            size=10),
            ybins=dict(  # bins used for histogram
                start=0.0,
                end=strip_max[f"{layer}V"],
                size=10))
        fig.update_layout(xaxis={"title": "Pos x [strips]"})
        fig.update_layout(yaxis={"title": "Pos v [strips]"})

        fig.write_html(f"{out_path}/cluster2Dpos{layer}.html", include_plotlyjs="../plotly.min.js")


        for view in ("x", "y"):
            if view == "y":
                view_string = "V"
            else:
                view_string = "X"

            fig = px.histogram(cluster2d_view, f"cl_charge_{view}", )
            fig.update_layout(template=standard_template,
                              title=f"Cluster charge (cluster2d) L{layer}{view_string} ")
            fig.update_traces(xbins=dict(  # bins used for histogram
                start=0.0,
                end=250.0,
                size=5
            ))
            fig.update_layout(yaxis={"title": "#Clusters"})
            fig.update_layout(xaxis={"title": "Cluster charge [fC]"})

            fig.write_html(f"{out_path}/cluster2D_charge{layer}{view_string}.html", include_plotlyjs="../plotly.min.js")


            ## Size1d

            fig = px.histogram(cluster2d_view, f"cl_size_{view}", )
            fig.update_layout(template=standard_template,
                              title=f"Cluster size (cluster2d) L{layer}{view_string} ")
            fig.update_traces(xbins=dict(  # bins used for histogram
                start=0.0,
                end=20.0,
                size=1
            ))
            fig.update_layout(yaxis={"title": "#Clusters"})
            fig.update_layout(xaxis={"title": "Cluster size [# strips]"})

            fig.write_html(f"{out_path}/cluster2D_size{layer}{view_string}.html", include_plotlyjs="../plotly.min.js")

