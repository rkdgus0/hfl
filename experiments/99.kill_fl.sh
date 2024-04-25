abc_host="etri@129.254.194.168"
bc_host="etri@129.254.194.168"
bc25_host="root@bc25" 
bc27_host="root@bc27"
kill_fl_process="kill -9 \`ps aux | grep 'python3 FL_*' | grep -v 'grep' | awk '{print \$2}'\`"

ssh $abc_host "$kill_fl_process"
ssh $bc25_host "$kill_fl_process"
ssh $bc27_host "$kill_fl_process"
