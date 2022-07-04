import os
import sys

run = sys.argv[1]
if len(sys.argv)>2:
    down_sampling = sys.argv[2]
else:
    down_sampling = 2

os.system(f"/civetta.py -d -a -c -c2 -cpu 30 {run} -down 20")
# ./civetta.py -ca_al $1 -cpu 35;
# ./civetta.py -d -a -c -c2 -cpu 30 $1 -down 2;
# ./civetta_perf.py $1 -perf -1 -sT 20 -sD 6 -chi -cpu 30;
# #./civetta_perf.py $1 -Ser;