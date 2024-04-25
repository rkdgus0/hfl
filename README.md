# ETRI ABC project Federated Learning Simulator
The simulator is Federated Learning (FL) simulator capable of single/hierarchical FL experiments.
It provides basic FL functions such as FL network configuration, users' data distribution settings, and FL parmeter settings
and provides statistical information such as real-time FL performance monitoring and blockchain-based FL history analysis.
It also provides additional functions such as performance evaluation method, user selection policy and model aggregation policy.

## Related Project
Project Name: ABC (MEC-Based AI-converged BlockChain)

Project Leader:  DongOh Kim (dokim@etri.re.kr)

Project Members: Heesang Jin (jinhs@etri.re.kr), JongDae Park (parkjd@etri.re.kr)

Project period:  2022.04.01 ~ 2024.12.31


## Requirements
We tested in the environment below.

Please match the software version below.

### Software
OS: Ubuntu 22.04.1

NVIDIA-driver: 515.65.01

CUDA: 11.7

cuDNN: 8.9.2

Anaconda: 22.9.0

Python: 3.10.6

Tensorflow: 2.12.0

### Hardware
CPU: 2x Intel Xeon Gold 6326 2.9G

SSD: 1.92 TB

Memory: 256GB

GPU: A40(46GB), P100(16GB), T4(16GB)

Network: 1Gb ethernet switch


## Getting started
Current main code is in 'RemoteVersion' directory.
```
$ cd RemoteVersion
```

Run an aggregator first.
```
$ python3 FL_Aggregator.py -num_total_users 4 -my_port 50081 -round 10
```

Run two usersets each which simulates two users, so total four users will begin.
```
$ python3 FL_UserSet.py -uset_index 0 -num_total_users 4 -num_users 2 -my_port 50101 -agg_port 50081
$ python3 FL_UserSet.py -uset_index 1 -num_total_users 4 -num_users 2 -my_port 50102 -agg_port 50081
```
Done!

If you increase the number of users to 12, set '-num_total_users' to '12' and set '-num_users' of both usersets to '6'.

Of course, you can also run 12 users by starting six userset processes each with '-num_users 2'. 

## FL information RPC server
Aggregator provides RPC server for getting FL information.

When FL is running, you should run the "interpreter" as bellow.
```
$ python3 interpreter.py -addr localhost:50081
```

Done!

## Execution parameters
### Aggregator (FL_Aggregator.py)
| Parameter          | Type   | Values          | Default      | Description                                                         |
|--------------------|--------|-----------------|--------------|---------------------------------------------------------------------|
| `-round`           | int    | 1~              | 10           | the number of MEC FL rounds                                         |
| `-fraction`        | float  | 0.0~1.0         | 1.0          | the fraction of selected users to request local training in the MEC |
| `-leader_ip`       | string | xxx.xxx.xxx.xxx | "localhost"  | IP of the leader to connect                                         |
| `-leader_port`     | string | xxxxx           | "50051"      | port number of the leader to connect                                |
| `-my_id`           | string |                 | "aggregator" | ID of the aggregator                                                |
| `-my_port`         | string |                 | "50081"      | port number of the aggregator                                       |
| `-hfl`             | bool   | True, False     | False        | ON/OFF for Hierarchical FL                                          |
| `-num_total_users` | int    | 1~              | 4            | the number users in the MEC (not userset)                           |


### Userset (FL_Userset.py)
| Parameter          | Type   | Values          | Default     | Description                                                                                    |
|--------------------|--------|-----------------|-------------|------------------------------------------------------------------------------------------------|
| `-epoch`           | int    | 1~              | 10          | the number of local training epochs                                                            |
| `-agg_ip`          | string | xxx.xxx.xxx.xxx | "localhost" | IP of the aggregator to connect                                                                |
| `-agg_port`        | string | xxxxx           | "50081"     | port of the aggregator to connect                                                              |
| `-num_mecs`        | int    | 1~              | 1           | the number of MECs (required for data distribution configuration)                              |
| `-mec_index`       | int    | 0~              | 0           | the index of the MEC connected with the userset (required for data distribution configuration) |
| `-my_ip`           | string | xxx.xxx.xxx.xxx | AUTO SETUP  | IP of the userset (not used)                                                                   |
| `-my_port`         | string | xxxxx           | "50101"     | port number of the userset                                                                     |
| `-user_index`      | int    | 0~              |             | the index of the userset in the MEC                                                            |
| `-num_total_users` | int    | 1~              | 4           | the number users in the MEC (not userset)                                                      |
| `-iid`             | bool   | True, False     | True        | whether user data distribution is iid or non-iid                                               |


## How to stop FL processes
Stop all running processes on the server 
```
$ kill -9 `ps aux | grep 'python3 FL_*' | grep -v 'grep' | awk '{print $2'}`
```

## Data distribution file generation
If you change the number of users and the number of training data they have,
you have to generate data distribution file corresponding to your settings.

If you want 8 users, go to 'datadist' directory.
```
$ cd datadist
```

Edit 'num_users_per_group' value to that you want to change in 'gen_dataset_distribution.py'.
```
num_users_per_group = VALUE_YOU_WANT # number of users contained in each MEC

```
Run the code.
```
$ python3 gen_dataset_distribution.py 
```

Check if 'dataset_distribution_1_8.json' was created.
```
$ ls
```

## Error: When multiple GPUs on server
If your server has multiple gpus, the following error can occur.

That's because the process uses 'set_memory_growth' in the tensorflow. 
```
ValueError: Memory growth cannot differ between GPU devices.
```

To handle the error, you have to assign a specific gpu number to the process like bellow.
```
$ CUDA_VISIBLE_DEVICES=0 python3 FL_Aggregator.py -num_total_users 4 -my_port 50081 -round 10
```


## Remote execution.
If you want to run multiple FL processes at once,

you should make use of script that can execute remotely.
```
$ cd tests
```

Check the contents of script files and run.
```
$ sh 1.small_fl.sh
```

FL processes are creating log files.

If you want to track logs of the FL processes,
use the 'tail' command at the server the FL processes is running. 
```
$ tail -f logs/aggregator.log
$ tail -f logs/userset1.log
$ tail -f logs/userset2.log
$ tail -f logs/userset3.log
```


## Authors
Heesang Jin (jinhs@etri.re.kr)

## License
For open source projects, say how it is licensed.

## Project status
On-going.