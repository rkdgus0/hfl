abc_host="etri@129.254.194.168"
bc25_host="root@bc25 conda activate fl;" 
bc27_host="root@bc27 conda activate fl;"

abc_root_dir="/home/etri/hsjin/abc_fl/fl/RemoteVersion"
bc_root_dir="/root/hsjin/abc_fl/fl/RemoteVersion"

gpu0="CUDA_VISIBLE_DEVICES=0"
gpu1="CUDA_VISIBLE_DEVICES=1"

is_iid="True"

# Aggregator
ssh $abc_host "cd $abc_root_dir; $gpu0 python3 FL_Aggregator.py -num_total_users 4 -my_port 50081 -round 100 &> logs/aggregator.log &"

# Userset 
ssh $abc_host "cd $abc_root_dir; $gpu0 python3 FL_UserSet.py -uset_index 0 -num_total_users 4 -num_users 1 -my_port 50101 -agg_ip 129.254.184.179 -agg_port 50081 -iid $is_iid &> logs/userset1.log &"
ssh $abc_host "cd $abc_root_dir; $gpu0 python3 FL_UserSet.py -uset_index 1 -num_total_users 4 -num_users 1 -my_port 50102 -agg_ip 129.254.184.179 -agg_port 50081 -iid $is_iid &> logs/userset2.log &"
ssh $abc_host "cd $abc_root_dir; $gpu1 python3 FL_UserSet.py -uset_index 2 -num_total_users 4 -num_users 1 -my_port 50103 -agg_ip 129.254.184.179 -agg_port 50081 -iid $is_iid &> logs/userset3.log &"
ssh $abc_host "cd $abc_root_dir; $gpu1 python3 FL_UserSet.py -uset_index 3 -num_total_users 4 -num_users 1 -my_port 50104 -agg_ip 129.254.184.179 -agg_port 50081 -iid $is_iid &> logs/userset4.log &"


############################# how to track the each log #############################
# $ tail -f logs/aggregator.log
# $ tail -f logs/userset1.log
# $ tail -f logs/userset2.log
# $ tail -f logs/userset3.log

############################# how to kill the processes #############################
# $ kill -9 `ps aux | grep "python3 FL_*" | grep -v "grep" | awk '{print $2}'`
