#!/bin/bash
echo "Run: $1";
echo "Downsampling: $2";
#echo "Full Name: $3";
./civetta.py -d -a -c -c2 -cpu 30 $1 -down 20;
./civetta.py -ca_al $1 -cpu 35;
./civetta.py -d -a -c -c2 -cpu 30 $1 -down 2;
./civetta_perf.py $2 -perf -1 -sT 20 -sD 6 -chi -cpu 30; ./civetta_perf.py $i -Ser;