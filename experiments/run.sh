#!/bin/bash

source activate hfl
cd ../

# Init Parameter
SLEEP_TIME=5

manager_address="10.20.22.107:8888"
config_name="fl_config_simul"
net_config_name="fl_net_config"
NUM_LEADER=1
NUM_AGG=1
NUM_USERSET=1


# FL_Manager.py 실행
echo "Starting HFL! (Leader: $NUM_LEADER, Aggregator: $NUM_AGG, Userset: $NUM_USERSET)"
python3 FL_Manager.py -m "$manager_address" -f "$net_config_name" &
sleep $SLEEP_TIME

# FL_Leader.py 실행
sleep $SLEEP_TIME
for ((i=0; i<$NUM_LEADER; i++))
do
    port=$((50000+i))
    python3 FL_Leader.py -i "leader$i" -p "$port" -m "$manager_address" &
done
sleep $SLEEP_TIME

# FL_Aggregator.py 실행
for ((i=0; i<$NUM_AGG; i++))
do
    port=$((50050+i))
    python3 FL_Aggregator.py -i "aggregator$i" -p "$port" -m "$manager_address" > /dev/null 2>&1 &
done
sleep $SLEEP_TIME

# FL_UserSet.py 실행
for ((i=0; i<$NUM_USERSET; i++))
do
    port=$((50100+n))
    python3 FL_UserSet.py -i "userset$i" -p "$port" -m "$manager_address" > /dev/null 2>&1 &
done
sleep $SLEEP_TIME

# FL_Client.py 실행
python3 FL_Client.py -m "$manager_address" -f "$config_name" &

wait
echo "Experiments completed!"