import datetime
import json
import pickle

import numpy as np
import tensorflow as tf
import os
import pandas as pd
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from matplotlib import pyplot as plt
from tensorflow.python.keras import Model, models, optimizers, losses, metrics
from sklearn.metrics import f1_score

MAX_MSG_LEN_256K = 262144
MAX_MSG_LEN_1M = 1048576
MAX_MSG_LEN_4M = 4194304
MAX_MSG_LEN_8M = 8388608
MAX_MSG_LEN_16M = 16777216  # 16 MB
MAX_MSG_LEN_32M = 33554432  # 32 MB
MAX_MSG_LEN_320M = 335544320 # 320 MB
MAX_MSG_LEN_960M = 1006632960  # 960 MB
MAX_MSG_LEN_1920M = 2013265920  # 1,920 MB
WANDB = False
WANDB_ID = ''
WANDB_API = ''
WANDB_PROJECT_NAME = 'ETRI_exp'
WANDB_PROJECT_ONLINE = 'online'
LOCAL_EVALUATE = False

MAX_MESSAGE_LENGTH = MAX_MSG_LEN_960M
MAX_MSG_LEN_NORMAL = MAX_MSG_LEN_16M
MAX_MSG_LEN_MODEL = MAX_MSG_LEN_960M

MANAGER_PORT = '9002'
AGGREGATOR_PORT = '50051'
USERSET_PORT = '50100'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', '') or v is None:
        return False

def default(type, input, default):
    return default if input in [None, '', ' '] else type(input)

def str_default(dictionary, key, default):
    return default if key not in dictionary or dictionary[key] is None or dictionary[key] == '' else dictionary[key]

def int_default(dictionary, key, default):
    return default if key not in dictionary or dictionary[key] is None or dictionary[key] == '' else int(dictionary[key])

def float_default(dictionary, key, default):
    return default if key not in dictionary or dictionary[key] is None or dictionary[key] == '' else float(dictionary[key])

def list_default(dictionary, key, default):
    return list(map(str, default)) if key not in dictionary or dictionary[key] is None or dictionary[key] == '' else list(map(str, dictionary[key].replace(' ', '').split(',')))
    #return list(map(str, (dictionary.get(key) or default).split(','))) if key not in dictionary or not dictionary[key] else list(map(str, dictionary[key].split(',')))

def bool_default(dictionary, key):
    return str2bool(dictionary[key]) if key in dictionary else False

def count_module(net_config, path, req_num, online=None):
    n_module = sum(1 for section in net_config.sections() if section.startswith(f'{path}'))
    on_module = sum(1 for section in net_config.sections() if section.startswith(f'{path}') and net_config[section].getint('current_used', 0) < online)
    if on_module < req_num:
        raise ValueError(f"[Manager] check the number of available {path}")
    if path == 'leader' and req_num > 1:
        raise ValueError("[Manager] check the number of collecting Leader")
    return n_module

def convert_proto_to_dict(proto_message):
    result_dict = {}
    for section_name, section_proto in proto_message.sections.items():
        result_dict[section_name] = {}
        for option_name, option_value in section_proto.options.items():
            result_dict[section_name][option_name] = option_value
    return result_dict

def make_tensorboard_dir(root_logdir):
    sub_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root_logdir, sub_dir_name)

def make_id(type, ip, port):
    ty = type.lower()
    if ty == 'globalaggregator' or ty == 'ga' or ty == 'leader':
        ty = 'ga'
    elif type == 'aggregator' or type == 'agg':
        ty = 'ag'
    elif type == 'userset' or type == 'us':
        ty = 'us'          
    
    idx = ip.rindex('.')
    id = ty + '_' + ip[idx+1:] + '_' + port
    return id

def serialize_param_service(leader, aggregator, user, userset, simul, end_condition):
    p_str = ''
    p_str = p_str + str(leader) + ',' + str(aggregator) + ',' + str(user) + ',' + str(userset)
    
    if simul:
        p_str = p_str + ',T'
    else:
        p_str = p_str + ',F'
    
    p_str = p_str + ',' + str(end_condition[0]) + ',' + str(end_condition[1])
    print(p_str)
    return p_str

def deserialize_param_service(p_str):
    p_strs = p_str.split(',')
    leader = int(p_strs[0])
    aggregator = int(p_strs[1])
    user = int(p_strs[2])
    userset = int(p_strs[3])
    print("@ leader={}, aggregator={}, user={}, userset={}".format(leader, aggregator, user, userset))
    
    if p_strs[4] == 'T': simul = True
    else: simul = False
    end_condition = [int(p_strs[5]), int(p_strs[6])]
    print("@ simul={}, end_condition={}".format(simul, end_condition))
    
    return leader, aggregator, user, userset, simul, end_condition

def serialize_param_agg(aggs, change, fraction):
    avail_agg_function = ['fedavg', 'fedat']
    
    n_agg = len(aggs)
    p_str = '' + str(n_agg)
    for agg in aggs:
        c_str = agg[0].lower()
        if any(str in c_str for str in avail_agg_function):
            p_str = p_str + ',' + agg[0] + ',' + str(agg[1])
        else:
            return None
        
    p_str = p_str + ',' + change[0] + ',' + str(change[1]) + ',' + str(fraction)
    print(p_str)    
    return p_str

def deserialize_param_agg(p_str):    
    p_strs = p_str.split(',')
    print(p_str)   
    num = int(p_strs[0]) 
    idx = 1
    print(num)    
    for n in range(num):
        agg = [p_strs[idx]]
        agg.append(int(p_strs[idx+1]))
        idx += 2
        if n == 0:
            aggs = [agg]
        else:
            aggs.append(agg)
            
    change = [p_strs[idx]]
    change.append(int(p_strs[idx+1]))
    idx += 2
    fraction = int(p_strs[idx])    
      
    return aggs, change, fraction

def deserialize_param_service(p_str):
    p_strs = p_str.split(',')
    leader = int(p_strs[0])
    aggregator = int(p_strs[1])
    user = int(p_strs[2])
    userset = int(p_strs[3])
    print("@ leader={}, aggregator={}, user={}, userset={}".format(leader, aggregator, user, userset))
    
    if p_strs[4] == 'T': simul = True
    else: simul = False
    end_condition = [int(p_strs[5]), int(p_strs[6])]
    print("@ simul={}, end_condition={}".format(simul, end_condition))
    
    return leader, aggregator, user, userset, simul, end_condition

#weight을 1차원 배열로 바꿔서 저장,배포
def serialize_model_weights(model_weights):
    layers = [] 
    layers_shape = [] 
    total_model_size = 0 
    for k, layer in enumerate(model_weights): # k는 index(0,1,2,...), layer는 model_weight의 값 
        # mcmahan2nn model_weights: (3072, 64), (64,), (64,10), (10,) 형태 -> 원래 코드에선 (784,64) , ~~였음
        delim = ',' 
        s = delim.join(map(str, layer.shape)) #s : layer의 shape을 str형태로 바꿔서 ,로 구분해 저장 -> 
        layers_shape.append(s) 
        layer_flat = layer.flatten()
        layer_proto = tf.make_tensor_proto(layer_flat)
        layer_pickle = pickle.dumps(layer_proto)
        size_estimate = len(layer_pickle)
        total_model_size += size_estimate
        layers.append(layer_pickle)
    return layers, layers_shape

def deserialize_model_weights(layers, layers_shape):
    model_weights = []
    for k, layer_pickle in enumerate(layers):
        layer_shape_str = layers_shape[k]
        layer_shape = tuple(map(int, layer_shape_str.split(',')))

        layer_proto = pickle.loads(layer_pickle)
        layer_flat = tf.make_ndarray(layer_proto)
        layer = layer_flat.reshape(layer_shape)

        model_weights.append(layer)
    return model_weights

def average_model(models, avg_ratio=None):
    new_weights = list()

    if avg_ratio is "Layer":
        for weights_list_tuple in zip(*models):
            sum_weights = sum(weights_list_tuple)  # 각 가중치의 합
            avg_weight = sum_weights / 20  # 전체 모델 수로 나눈 평균
            new_weights.append(avg_weight)
    elif avg_ratio is None:
        for weights_list_tuple in zip(*models):
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
    elif avg_ratio is not None:
        for weights_list_tuple in zip(*models):
            new_weights.append(
                np.array([np.average(np.array(w), weights=avg_ratio, axis=0) 
                          for w in zip(*weights_list_tuple)])
            )
    return new_weights


def make_request_id(id, fl_round):
    return f"{id}_{fl_round}"


#Aggregation method
def test(test_data, global_model):
    test_x, test_y = test_data
    test_y = np.argmax(test_y, axis=1)
    return global_model.evaluate(test_x, test_y, verbose = 0)
        
def f1_test(test_data, global_model, agg_method):
    test_x, test_y = test_data
    pred_y = global_model(test_x)
    pred_y = np.argmax(pred_y, axis=1).astype(int)
    test_y = np.argmax(test_y, axis=1).astype(int)
    if 'macro' in agg_method:
        f1_scores = f1_score(test_y, pred_y, average='macro')
    else:
        f1_scores = f1_score(test_y, pred_y, average='micro')
    return f1_scores

def calc_avg_ratio(test_data, global_model, method, models, participating_mec, num_aggs_data):
    ratio = []
    ratio_avg = 0
    method = method.lower()
    if method == 'acc':
        for model in models:
            global_model.set_weights(model)
            _, acc = test(test_data, global_model)
            ratio.append(acc)
            ratio_avg += acc
    elif 'f1' in method:
        for model in models:
            global_model.set_weights(model)
            f1_scores = f1_test(test_data, global_model, method)
            ratio.append(f1_scores)
            ratio_avg += f1_scores
    elif method == 'fedavg':
        for mec_idx in participating_mec:
            ratio.append(num_aggs_data[mec_idx])
            ratio_avg += num_aggs_data[mec_idx]
    elif method == 'equal':
        ratio = [1]*len(models)
        ratio_avg += len(models)
    else:
        raise ValueError('Invalid Aggregation Method!')
    #==========================아직 구현 X delay관련 파라미터가 없음========================
    '''elif method == 'FedAT':
        at_ratio = [1 + x for x in list(self.init_mec_delay)]
        at_ratio *= self.delay_epoch + 1
        for mec_idx in participating_mec:
            ratio.append(at_ratio[mec_idx])
            ratio_avg += at_ratio[mec_idx]'''
    #=====================================================================================
    #print(f"[FL_Leader 모델 가중치 합산 확인] Method : {method}, Ratio: {[x / ratio_avg for x in ratio]}")
    return [x / ratio_avg for x in ratio]

#안쓰는듯
def get_num_per_labels(y_train):
    df = pd.DataFrame(y_train)
    idxes = [idx[0] for idx in df.value_counts().sort_index().index.tolist()]
    values = [value for value in df.value_counts().sort_index().values.tolist()]
    num_data_per_labels = []
    j = 0
    for i in range(0, 10):
        if i in idxes:
            num_data_per_labels.append(values[j])
            j += 1
        else:
            num_data_per_labels.append(0)
    return num_data_per_labels

