# Getting started

```
conda create -n hfl python=3.10.6
conda activate hfl
git clone https://github.com/rkdgus0/hfl.git
cd hfl
pip install -r tf_requirements.txt
pip install -r requirements.txt
bash experiments/run.sh
```

All the log will be saved in ```Logs``` folder.

## Execution parameters
### Aggregator (fl_config.ini)
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

## License
For open source projects, say how it is licensed.

## Project status
On-going.
