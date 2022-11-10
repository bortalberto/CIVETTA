import os
import sys

run = sys.argv[1]
if len(sys.argv)>2:
    down_sampling = sys.argv[2]
else:
    down_sampling = 2

os.system(f"./civetta.py -d -a -c -c2 -cpu 30 {run} -down 20")
os.system(f"./civetta.py -ca_al {run} -cpu 35")
os.system(f"./civetta.py -d -a -c -c2 -cpu 30 {run} -down {down_sampling}")
os.system(f"./civetta_perf.py {run} -perf -sT 20 -sD 6 -chi -cpu 20;")
os.system(f"./civetta_perf.py {run} -Ser;")

