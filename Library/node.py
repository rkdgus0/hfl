import numpy as np
import time
import socket
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from scipy.sparse import csr_matrix
import os

class FL_Node:
    '''
    FL_Node의 공통 기능 관련 클래스 
    '''
    def __init__(self, flconfig_file_path):
        self.node_type = "SLAVE"
        self.node_id = "SLAVE"
        self.node_ip = socket.gethostbyname(socket.gethostname())
        self.sleep_node = 1
        self.node_area = "MEC1"
        #self.role = ""
    
    def get_weight_difference(self, global_weights, private_model):
        file_name = "gradient.npz"
        
        private_weights = private_model.get_weights()
        weight_differences = [np.subtract(g_weight, p_weight) 
                              for g_weight, p_weight in zip(global_weights, private_weights)]
        
        return weight_differences
    
    def save_gradient(self, gradient, top_n_percent):
        file_name = "gradient.npz"
                
        sparse_gradient = []
        for grad in gradient:
            threshold = np.percentile(np.abs(grad), 100 - top_n_percent)
            abs_grad = np.abs(grad)
            grad[abs_grad <= threshold] = 0

            reshaped_grad = grad.reshape(-1, grad.shape[-1])
            sparse_grad = csr_matrix(reshaped_grad)
            sparse_gradient.append(sparse_grad)

        sparse_gradient_file = {}
        for i, sparse_grad in enumerate(sparse_gradient):
            sparse_gradient_file[f"matrix_{i}_data"] = sparse_grad.data
            sparse_gradient_file[f"matrix_{i}_indices"] = sparse_grad.indices
            sparse_gradient_file[f"matrix_{i}_indptr"] = sparse_grad.indptr
            sparse_gradient_file[f"matrix_{i}_shape"] = sparse_grad.shape

        np.savez(file_name, **sparse_gradient_file)
        return file_name   
    
    def count_data(self, dataset):
        class_counts = np.sum(dataset['y_train'], axis=0).astype(int)
        data_num = [len(dataset['y_train'])] + list(class_counts)
        return data_num 
    
    def compare_model(model1, model2, print_line=2):
        print("---compare_model()---")
        rows = 0
        fault = 0
        same = 0
        same_zeros = 0
        
        layers1 = model1.get_weights()
        layers2 = model2.get_weights()
        
        for layer1, layer2 in zip(layers1, layers2):
            if isinstance(layer1, np.ndarray) and isinstance(layer2, np.ndarray):
                layer1_flat = np.ravel(layer1, order='C')
                layer2_flat = np.ravel(layer2, order='C')
                for e1, e2 in zip(layer1_flat, layer2_flat):
                    if rows < print_line:
                        print("[{}] {} {}".format(rows, e1, e2))
                    if e1 != e2:
                        fault += 1
                    else:
                        if same < print_line:
                            print("[{}] {} {}".format(rows, e1, e2))
                        same += 1
                        if e1 == 0:
                            same_zeros += 1
                    rows += 1
            else:
                print("Skipping layer: ", layer1, layer2)
        
        print("@ Rows: {}, Same: {}(Zeros: {}), Differ: {}".format(rows, same, same_zeros, fault))

    # 특정 레이어를 제외한 나머지 레이어를 0으로 설정
    # Userset에서 호출하여 사용
    # (mcmahan2nn 기준 (0,1)(2,3)(4,5)로 묶음)
    # 1, 3, 5 layer는 bias Layer
    def set_weights_zero_except_one_layer(model, target_layer_index):
        # Get the model's weights
        weights = model.get_weights()
        new_weights = []

        for i, layer_weights in enumerate(weights):
            # Check if the current layer is the target layer
            if target_layer_index * 2 <= i <= target_layer_index * 2 + 1:
                new_weights.append(layer_weights)
            else:
                # If it is not the target layer, set the weights to zeros   
                new_weights.append(np.zeros_like(layer_weights))

        model.set_weights(new_weights)
        return model
    
    # TODO
    # 라운드별 Userset 인원에 대한 선택된 상위 가중치를 누적으로 카운트 하는 함수 
    # 추후에 aggregator에서 취합하여 가중치 합/ 선택횟수

    def accumulate_occurrences(occurrences_list, total_occurrences=None):
        if total_occurrences is None:
            total_occurrences = np.zeros_like(occurrences_list[0])

        for occurrences in occurrences_list:
            total_occurrences += occurrences
        return total_occurrences
    
    def count_top_weight_occurrences_across_models(weight_differences_list, top_percent=0.1):
        # 모든 가중치의 차이를 하나의 배열로 병합
        flat_weight_diff_all = np.concatenate([diff.flatten() for diff in weight_differences_list])

        # 상위 퍼센트에 해당하는 가중치의 인덱스를 찾음
        sorted_indices = np.argsort(np.abs(flat_weight_diff_all))[::-1]
        threshold_index = int(len(sorted_indices) * top_percent)
        top_indices_all = sorted_indices[:threshold_index]

        # 각 가중치의 위치별로 누적으로 선택된 횟수 계산
        num_weights = flat_weight_diff_all.shape[0]
        occurrences = np.zeros(num_weights, dtype=int)
        occurrences[top_indices_all] += 1

        return occurrences
    

    # input : model , weight_differences,percent (가중치 변화량, 퍼센트)
    # output : 상위 퍼센트 가중치를 제외하고 나머지 0으로 설정한 모델

    def retain_top_weights(model, weight_differences,percent):
        # 가중치 평탄화
        flat_differences = np.concatenate([arr.flatten() for arr in weight_differences])
        
        # 10% 선별
        sorted_indices = np.argsort(np.abs(flat_differences))
        top_10_percent_idx = sorted_indices[-int(len(sorted_indices) * percent):]
        
        # 각 층 레이어 인덱스 재구성
        layer_shapes = [arr.shape for arr in weight_differences]
        layer_indices = []
        start_idx = 0
        for shape in layer_shapes:
            layer_indices.append((start_idx, start_idx + np.prod(shape)))
            start_idx += np.prod(shape)
        
        # 나머지는 0으로
        new_weights = []
        for layer, (start, end) in enumerate(layer_indices):
            weights = model.get_weights()[layer]
            flat_weights = weights.flatten()
            for idx, val in enumerate(flat_weights):
                if idx + start in top_10_percent_idx:
                    continue
                flat_weights[idx] = 0.0
            new_weights.append(flat_weights.reshape(weights.shape))
        
        # 가중치 기반 모델 적용
        model.set_weights(new_weights)
        return model
        
def save_model(model, path):
    '''
    지정 모델 저장 
    '''
    print("@ call save_model() ------")
    path = 'my_model.h5'
    # Save the entire model to a HDF5 file
    model.save(path)
    model_size = len(model)/1024
    print("@ save model path: {}, size: {}".format(path, model_size))    

def load_model(path):
    '''
    지정 모델 불러오기
    '''
    try:
        print("@ call load_model() ------")
        path = 'my_model.h5'
        loaded_model = tf.keras.models.load_model(path)
        model_size = len(loaded_model)/1024    
        print("@ load model path: {}, size: {}".format(path, model_size))    
        loaded_model.summary()
    except:       
        print('예외가 발생했습니다.') 
        return None
    
    return loaded_model
    
def show_model(model, print_line=2):
    '''
    지정 모델 가중치 값 확인 함수 
    '''
    print("---show_model()---")
    rows = 0; zeros = 0; max_w =0; min_w = 0
    layers = np.ravel(model.get_weights(), order='C')
    for layer in layers:
        for e in np.ravel(layer, order='C'):
            if rows < print_line: print("[{}] {}".format(rows, e))
            if e == 0: zeros += 1
            if e > max_w: max_w = e
            if e < min_w: min_w = e
            rows += 1
    print("@ Rows: {}, Zeros: {}, Min: {}, Max: {}".format(rows, zeros, min_w, max_w))



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

s = serialize_param_agg([['FedAvg', 100]], ['test', 50], 100)
print(s)
aggs, change, fraction = deserialize_param_agg('2,FedAvg,50,FedAT,50,test,50,100') 
print(aggs)
print(change)
print(fraction)

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
     
def model_predict(idx, x_test, y_test):
    '''
    모델 추론 활용
    Generates output predictions for the input samples.
    '''
    print("----- model_predict()---------")
    if idx > 10000: print("error idx is too large"); return
    test_idx = idx
    x_test_p = x_test[test_idx:test_idx+1]
    y_t_v = y_test[test_idx:test_idx+1]
    x_t_v = model.predict(x_test_p)
    print("@ answer: [{}], {}".format(np.argmax(y_t_v), y_t_v))
    print("@ predict:[{}], {}".format(np.argmax(x_t_v), np.round(x_t_v,0)))
    
def save_model_weight(model, path):
    '''
    지정 모델 저장 
    '''
    print("@ call save_model_weight() ------")
    path = 'my_model_weight'
    # Save the entire model to a HDF5 file
    model.save_weights(path)
    print("@ save model weight path: {}".format(path))    

def load_model_weight(path):
    '''
    지정 모델 불러오기
    '''
    print("@ call load_model_weight() ------")
    path = 'my_model_weight'
    loaded_model_weight = tf.keras.models.load_weights(path)
    print("@ load model weight path: {}".format(path))    
    
    return loaded_model_weight

