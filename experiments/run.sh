#!/bin/bash

source activate hfl
cd ../

# Init Parameter
SLEEP_TIME=5
manager_address="10.20.22.107:8888"
config_name="fl_config_simul"
net_config_name="fl_net_config"

# Num of Component
NUM_LEADER=$(grep -Po "(?<=leader = ).*" "config/$config_name.ini")
NUM_AGG=$(grep -Po "(?<=aggregator = ).*" "config/$config_name.ini")
NUM_USERSET=$(grep -Po "(?<=userset = ).*" "config/$config_name.ini")

# FL_Manager.py 실행
echo "Starting HFL! (Leader: $NUM_LEADER, Aggregator: $NUM_AGG, Userset: $NUM_USERSET)"
python3 FL_Manager.py -m "$manager_address" -f "$net_config_name" 1> Logs/manager.log 2>&1 &
sleep $SLEEP_TIME

# FL_Leader.py 실행
sleep $SLEEP_TIME
for ((i=0; i<$NUM_LEADER; i++))
do
    port=$((50000+i))
    python3 FL_Leader.py -i "leader$i" -p "$port" -m "$manager_address" 1> Logs/leader$i.log 2>&1 &
done
sleep $SLEEP_TIME

# FL_Aggregator.py 실행
for ((i=0; i<$NUM_AGG; i++))
do
    port=$((50050+i))
    python3 FL_Aggregator.py -i "aggregator$i" -p "$port" -m "$manager_address" 1> Logs/aggregator$i.log 2>&1 &
done
sleep $SLEEP_TIME

# FL_UserSet.py 실행
for ((i=0; i<$NUM_USERSET; i++))
do
    port=$((50100+n))
    python3 FL_UserSet.py -i "userset$i" -p "$port" -m "$manager_address" 1> Logs/userset$i.log 2>&1 &
done
sleep $SLEEP_TIME

# FL_Client.py 실행
python3 FL_Client.py -m "$manager_address" -f "$config_name" 1> Logs/client.log 2>&1&

wait
echo "Experiments completed!"