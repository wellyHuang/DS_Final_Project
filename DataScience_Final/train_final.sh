#! /bin/sh
A=0
B=0
A=$(date +%s)
python3 datascience_final_r05921083.py $1
B=$(date +%s)
echo $((B-A))