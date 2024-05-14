host="gang@10.20.22.107"
port="55125"
pasword="mill"
kill_cmd="kill -9 \`ps aux | grep 'python3 FL_*' | grep -v 'grep' | awk '{print \$2}'\`"

sshpass -p $password ssh $host -p $port "$kill_cmd"