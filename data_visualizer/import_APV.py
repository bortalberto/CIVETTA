import configparser
import sys
import os
import root_pandas
import pandas as pd
config = configparser.ConfigParser()
config.read("../config.ini")
from tqdm import tqdm
try:
    data_folder=config["GLOBAL"].get("APV_data_folder")
    data_folder_out=config["GLOBAL"].get("CIVETTA_APV_folder")

except KeyError as E:
    print (f"{E}Missing or partial configration file, restore it.")
    sys.exit(1)
if len(sys.argv)!=2:
    print ("Insert only APV run number")
    sys.exit(1)
run_number=int(sys.argv[1])
data_path_to_analyze=f"{data_folder}/evt_run{run_number}.root"
if not os.path.isfile(data_path_to_analyze):
    print (f"Can't find {data_path_to_analyze}")
    sys.exit(1)
print (f"Converting run {run_number} for visualization")
print ("Loading files")
## Building hits_pd
APV_pd = root_pandas.read_root(data_path_to_analyze)
print ("Converting")
hit_keys=[key for key in APV_pd.keys() if ("Hit_" in key and all( [a_string not in key for a_string in ["nHit", "Cluster"] ]))]
cluster_keys=[key for key in APV_pd.keys() if "GemCluster1d_" in key]
cluster_keys.remove("GemCluster1d_nCluster")

dict_4_pd_hit={}
dict_4_pd_cl={}

dict_4_pd_hit["trigger_event"]=[]
dict_4_pd_cl["trigger_event"]=[]

for key in hit_keys:
    dict_4_pd_hit[key]=[]
for key in cluster_keys:
    dict_4_pd_cl[key] = []


for row in tqdm(APV_pd.iterrows(), total=len(APV_pd)):
    for i in range (0,len(row[1].GemHit_ID)):
        dict_4_pd_hit["trigger_event"].append(row[1]["trigger_event"])
        for key in hit_keys:
            dict_4_pd_hit[key].append(row[1][key][i])
    for i in range (0,len(row[1].GemCluster1d_ID)):
        dict_4_pd_cl["trigger_event"].append(row[1]["trigger_event"])
        for key in cluster_keys:
            dict_4_pd_cl[key].append(row[1][key][i])


hit_pd_APV=pd.DataFrame(dict_4_pd_hit)
cluster_pd_APV=pd.DataFrame(dict_4_pd_cl)
print ("Saving")

hit_pd_APV.to_pickle(os.path.join(data_folder_out,f"APV_run_{run_number}_hits.gzip"), compression="gzip")
cluster_pd_APV.to_pickle(os.path.join(data_folder_out,f"APV_run_{run_number}_cl.gzip"), compression="gzip")
