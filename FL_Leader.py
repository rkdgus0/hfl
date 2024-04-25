import argparse
#import configparser
import copy
from datetime import datetime
import logging
import os
import pickle
import random
import socket
import threading
import time
from concurrent import futures

from sklearn.utils import shuffle
import google
import grpc
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models, losses, metrics
from sklearn.metrics import f1_score

import fl_pb2
import fl_pb2_grpc

import Library.helper as helper
from Library.datasets import *
from Library.models import *
from Library.ipfs import FL_IPFS
from Library.UNIST import SERVER, MEC, USER

import wandb

MAX_MESSAGE_LENGTH = helper.MAX_MESSAGE_LENGTH
# tensorboard
tensorboard_log_dir = "tb_logs"

# ======= Leader의 Rpc통신 =========
class LeaderRpcService(fl_pb2_grpc.Leader):
    def __init__(self, leader):
        super().__init__()
        self.leader: FlLeader = leader
    
    # Aggregator의 id/ip/port를 가져와서 register_aggregator 딕셔너리에 저장
    def RegisterAggregator(self, addr_msg, context): 
        print(
            f"[Leader] Aggregator Registration. ID: {addr_msg.id}, IP: {addr_msg.ip}, Port: {addr_msg.port}")
        new_aggregator = {'id': addr_msg.id, 'ip': addr_msg.ip, 'port': addr_msg.port}
        self.leader.registered_aggregators[addr_msg.id] = new_aggregator

        my_addr = self.leader.get_pb2_address()
        return fl_pb2.ReturnMsg(accepted=True, sender_addr=my_addr)

    def ReturnMecModel(self, model_msg, context):
        fl_round = model_msg.fl_round
        aggregator_id = model_msg.sender_addr.id
        num_aggs_data = model_msg.num_trained_data
        request_id = helper.make_request_id(aggregator_id, fl_round)

        if len(self.leader.num_aggs_data) < self.leader.num_target_collecting: 
            self.leader.num_aggs_data.append(num_aggs_data)
            self.leader.num_total_data += num_aggs_data
        print(f"[Leader] [{fl_round} Round] Received the mec model from aggregator: {aggregator_id}, request: {request_id} ")
        self.leader.lock_aggregator_waiting_queue.acquire()
        request_id_exist = request_id in self.leader.aggregator_waiting_queue

        # ipfs == True 、 download model weight 
        if flLeader.ipfs ==True:
            model_hash = model_msg.model_hash
            print(f"[IPFS] aggregator's weight hash출력: {model_hash}") 
            downloaded_model_file_name = ipfs.download_from_ipfs(model_hash)

            if request_id_exist:
                self.leader.register_mec_model_IPFS(model_msg,downloaded_model_file_name)
                self.leader.aggregator_waiting_queue.remove(request_id)
                os.remove(downloaded_model_file_name)
        # ipfs ==False
        else:
            if request_id_exist:
                self.leader.register_mec_model(model_msg)
                self.leader.aggregator_waiting_queue.remove(request_id)
            
        
        print(f"[Leader] [{fl_round} Round] Remove request: {request_id}, "f"Len of queue: {len(self.leader.aggregator_waiting_queue)} ")    
        self.leader.lock_aggregator_waiting_queue.release()
        return fl_pb2.ReturnMsg(fl_round=fl_round, accepted=True, num_trained_data=model_msg.num_trained_data)
    
    # Function to get Leader/Simulator Parameter from Manager 
    def ConfigSending(self, request, context):
        config_dict = helper.convert_proto_to_dict(request)

        FL_Service = config_dict['FL_Service']
        Aggregation_Method = config_dict['Aggregation_Method']
        # ====== Participate_Info ======
        self.leader.simulate = helper.bool_default(FL_Service, 'simulate')
        end_condition = helper.list_default(FL_Service, 'end_condition', '5, 99')
        self.leader.global_round = int(end_condition[0])
        self.leader.stop_acc = float(end_condition[1])
        self.leader.num_user = helper.int_default(FL_Service, 'user', 100)
        self.leader.num_init_aggregators = helper.int_default(FL_Service, 'aggregator', 1)
        self.leader.num_target_collecting = helper.int_default(FL_Service, 'aggregator', 1)
        self.leader.ipfs = helper.bool_default(FL_Service, 'ipfs')
        
        # ====== Wandb_Info ====== wandb 라이센스 이슈로 인해 주석처리(helper에서 hardcoding으로 변환)
        #self.leader.wandb = helper.bool_default(FL_Service, 'wandb')
        #self.leader.wandb_id = helper.str_default(FL_Service, 'wandb_id', None)
        #self.leader.wandb_api = helper.str_default(FL_Service, 'wandb_api', None)
        self.leader.exp_name = helper.str_default(FL_Service, 'exp_name', 'exp')
        self.leader.group_name = helper.str_default(FL_Service, 'group_name', 'EXP')

        # ====== Model/Data_Info ======
        self.leader.model = helper.str_default(FL_Service, 'model', 'mcmahan2NN')
        self.leader.pretrained = helper.bool_default(FL_Service, 'pretrained')
        self.leader.optimizer = helper.str_default(FL_Service, 'optimizer', 'adam')
        self.leader.fraction = helper.int_default(FL_Service, 'frac', 100)
        self.leader.base_model = define_model(self.leader.model , self.leader.pretrained)
        self.leader.dataset = helper.str_default(FL_Service, 'dataset', 'mnist')
        (x_train, y_train), self.leader.test_data = load_dataset(self.leader.dataset)

        # ====== Aggregation_method_Info ======
        self.leader.hybrid = helper.bool_default(Aggregation_Method, 'hybrid')
        self.leader.total_agg_method = helper.list_default(Aggregation_Method, 'agg_method_1', 'Fedavg')
        agg_method = helper.list_default(Aggregation_Method, 'agg_method', 'Fedavg, 100')
        add_agg_method1 = helper.list_default(Aggregation_Method, 'add_agg_method1', ',')
        add_agg_method2 = helper.list_default(Aggregation_Method, 'add_agg_method2', ',')
        self.leader.agg_method = agg_method[0]
        self.leader.basic_method = agg_method[0]
        self.leader.agg_param = int(agg_method[1])
        self.leader.add_agg_method1 = helper.default(str, add_agg_method1[0], None)
        self.leader.add_agg_param1 = helper.default(int, add_agg_method1[1], 30)
        self.leader.add_agg_method2 = helper.default(str, add_agg_method2[0], None)
        self.leader.add_agg_param2 = helper.default(int, add_agg_method2[1], 30)
        self.leader.adaptive_agg_method = helper.str_default(Aggregation_Method, 'adaptive_agg_method', 'no_adapt')
        self.leader.adaptive_parameter = helper.int_default(Aggregation_Method, 'adaptive_parameter', 40)
        self.leader.eval_batch = helper.int_default(Aggregation_Method, 'evaluation_batch_size', 64)

        # ====== Simulator_Info ======
        # Only when the simulate parameter is True
        if self.leader.simulate:
            self.leader.split = helper.str_default(FL_Service, 'data_split', 'iid')
            self.leader.alpha = helper.float_default(FL_Service, 'diri_alpha', 3)

            self.leader.local_epoch = helper.int_default(FL_Service, 'local_epoch', 5)
            self.leader.local_batch_size = helper.int_default(FL_Service, 'local_batch_size', 256)
            self.leader.learning_rate = helper.float_default(FL_Service, 'local_learning_rate', 0.001)
            self.leader.lr_decay = helper.float_default(FL_Service, 'learning_rate_decay', 0.99)
            self.leader.lr_decay_round = helper.int_default(FL_Service, 'learning_rate_decay_round', 10)

            agg_delay = helper.list_default(FL_Service, 'agg_delays', '0,0,0,0,1,1,2,2,3,3')
            self.leader.agg_delay = list(map(int, agg_delay))
            self.leader.classification = helper.str_default(FL_Service, 'classification', None)
            self.leader.user_mapping = helper.str_default(FL_Service, 'user_mapping', 'equal')
            self.leader.delay_method = helper.str_default(FL_Service, 'delay_method', 'Range')
            self.leader.delay_range = helper.int_default(FL_Service, 'delay_range', 2)
            self.leader.delay_epoch = helper.int_default(FL_Service, 'delay_epoch', 0)
            self.leader.eval_every = helper.int_default(FL_Service, 'eval_every', 1)
            self.leader.model_decay = helper.str_default(FL_Service, 'model_decay', 'Equal')
            self.leader.x_dataset, self.leader.y_dataset = split_data_set(self.leader.split, x_train, y_train, self.leader.num_user, self.leader.alpha)

            print('[Leader] we get the Config Datas for Simulator!')

        return google.protobuf.empty_pb2.Empty()

class FlLeader:
    def __init__(self):
        self.registered_aggregators = {}
        self.address = {}
        self.selected_aggregators = []  # sampled users to request local training
        self.collected_aggregators = []  # users who have sent user model in current round
        self.num_aggs_data = []
        self.lock_aggregator_waiting_queue = threading.Lock()
        self.aggregator_waiting_queue = set()
        self.global_model = None
        self.base_model = None
        self.test_data = None
        self.eval_batch = None
        self.num_total_data = 0
        self.x_dataset = None
        self.y_dataset = None
        self.ipfs = None
        self.simulate = None
        self.acc = 0
    
    # ====== Server compose (Simulator용 함수) ======
    def compose_server(self, mecs):
        num_mec_datas = []
        for mec_mapping in mecs.mec_user_mapping.values():
            num_mec_data = 0
            for client_idx in mec_mapping:
                num_mec_data += len(mecs.clients.x_dataset[client_idx])
            num_mec_datas.append(num_mec_data)

        return SERVER(self.base_model, mecs, self.num_target_collecting, self.agg_delay,
                      self.test_data, self.delay_method, self.eval_batch, self.hybrid, 
                      self.delay_range, self.total_agg_method, self.delay_epoch, 
                      self.local_epoch ,self.model_decay, num_mec_datas)
    
    # ====== Aggregator compose (Simulator용 함수) ======
    def compose_mec(self, edges):
        NUM_MEC = self.num_target_collecting
        NUM_CLIENT = self.num_user
        mec_user_mapping = dict()
        clients = list(range(NUM_CLIENT))
        shuffle(clients)

        # User Grouping Function
        # user_mapping : aggregator별로 가져갈 유저 수(equal, diff)
        # Classification(n_data, n_class) : 많은 것끼리, 적은 것끼리 그룹
        # TODO Aggregator별로 데이터 수와 클래스 개수를 비슷하게 가지도록 그룹화
        if self.user_mapping.lower() == 'equal':
            if self.classification == 'n_data' or self.classification == 'n_class':
                num_client_per_mec = NUM_CLIENT // NUM_MEC
                for mec_idx in range(NUM_MEC-1):
                    mec_user_mapping[mec_idx] = self.mapping[mec_idx * num_client_per_mec : (mec_idx+1) * num_client_per_mec]
                mec_user_mapping[NUM_MEC-1] = self.mapping[(NUM_MEC-1) * num_client_per_mec : ]
                print(f'MEC_매핑 : {mec_user_mapping}')
            else:
                num_client_per_mec = NUM_CLIENT // NUM_MEC
                for mec_idx in range(NUM_MEC):
                    mec_user_mapping[mec_idx] = np.random.choice(range(NUM_CLIENT), num_client_per_mec, replace=False)
        elif self.user_mapping.lower() == 'diff':
            user_mapping = [[] for _ in range(NUM_MEC)]
            for i in range(NUM_CLIENT):
                mec_idx = random.randint(0, NUM_MEC - 1)
                user_mapping[mec_idx].append(i)
                user_mapping = sorted(user_mapping, key=len)
            for mec_idx, num_client in enumerate(user_mapping):
                print(f"MEC {mec_idx}: {num_client}")
                mec_user_mapping[mec_idx] = user_mapping[mec_idx]

        return MEC(self.base_model, mec_user_mapping, edges)

    # ====== User compose (Simulator용 함수) ======
    def compose_user(self, x_dataset, y_dataset):
        client_index = []
        NUM_CLIENT = self.num_user
        if self.classification == 'n_data':
            for user_idx in range(NUM_CLIENT):
                user_data = len(y_dataset[user_idx])
                client_index.append(user_data)
        elif self.classification == 'n_class':
            for user_idx in range(NUM_CLIENT):
                user_data = y_dataset[user_idx]
                tmp_y = np.argmax(user_data, axis=1)
                unique_labels = len(np.unique(tmp_y, return_counts=False))
                client_index.append(unique_labels)
        print(f'전처리 전 Client 매핑 : {client_index}')
        client_index = list(enumerate(client_index))
        client_index = sorted(client_index, key=lambda x: x[1])
        self.mapping = [index for index, _ in client_index]
        print(f'Mapping 데이터 : {self.mapping}')
        return USER(lr=self.learning_rate, model=self.base_model, x_dataset=x_dataset, y_dataset=y_dataset,
                    epochs=self.local_epoch, batch_size=self.local_batch_size)

    # Parameter 적용 확인 함수
    def check_config_file(self):
        while self.eval_batch is None:
            print("[Leader] Waiting for input parameter to be present by Client...")
            time.sleep(5)
    
    # id/ip/port 정보 mapping
    def set_address(self, _id, _ip, _port):
        self.address['id'] = _id
        self.address['ip'] = _ip
        self.address['port'] = _port

    def get_pb2_address(self):
        _id = self.address['id']
        _ip = self.address['ip']
        _port = self.address['port']

        return fl_pb2.AddressMsg(id=_id, ip=_ip, port=_port)

    # 리더에 aggregator 들어올 경우, 저장하는 함수
    def register_aggregator(self, message):
        new_aggregator = {'id': message.id, 'ip': message.ip, 'port': message.port}
        self.registered_aggregators[message.id] = new_aggregator

    # Aggregator의 모델을 IPFS에 저장하는 함수 (IPFS 겼을때 사용)
    def register_mec_model_IPFS(self, message,model_weight_IPFS):
        model_weights_IPFS = FL_IPFS.load_model_weights_from_ipfs(model_weight_IPFS)
        fl_round = message.fl_round
        aggregator_id = message.sender_addr.id
        _aggregator = self.registered_aggregators[aggregator_id]
        _aggregator[f"{fl_round}"] = model_weights_IPFS

    # Aggregator의 모델을 로컬(리더)에 저장하는 함수 (IPFS 껐을때 사용)
    def register_mec_model(self, message):
        model_weights = helper.deserialize_model_weights(message.model_layers, message.model_layers_shape)
        fl_round = message.fl_round
        aggregator_id = message.sender_addr.id
        _aggregator = self.registered_aggregators[aggregator_id]
        _aggregator[f"{fl_round}"] = model_weights

    # ===== RPC server =====
    def rpc_serve(self):
        my_port =  self.address['port']
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                             options=[
                                 ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                 ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                 ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                             ])
        rpcService = LeaderRpcService(self)
        fl_pb2_grpc.add_LeaderServicer_to_server(rpcService, server)
        server.add_insecure_port(f'[::]:{my_port}')

        server.start()
        print(f"[Leader] Rpc service started!, listening on {my_port}")
        self.grpc_server = server
        # server.wait_for_termination()

    # ===== Manager's RPC 함수 호출 / Manager에게 자신을 등록하는 함수
    def register_Leader_to_manager(self):
        print(f"[Leader] Send Leader registration request to the manager ({manager_ip}:{manager_port})")
        with grpc.insecure_channel(f'{manager_ip}:{manager_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.ManagerStub(channel)
            my_addr = flLeader.get_pb2_address()
            ret_msg = stub.RegisterLeader(my_addr)

            print(f"[Leader] Registration result: {ret_msg.accepted}")

    def send_final_results_to_manager(self, g_round, global_accs, global_losses, num_agg, elapsed_time):
        with grpc.insecure_channel(f'{manager_ip}:{manager_port}',
                                options=[
                                    ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                ]) as channel:
            stub = fl_pb2_grpc.ManagerStub(channel)
            
            final_results_msg = fl_pb2.FinalResultsMsg(
                g_round=g_round,
                accuracy=global_accs,
                loss=global_losses,
                num_aggs=num_agg,
                elapsed_time=elapsed_time
            )
            
            ret_msg = stub.ReportFinalResults(final_results_msg)
            print(f"[Leader] Final results sent to Manager. Result: {ret_msg.accepted}")

    # Leader의 현재 상태(online/offline) 정보를 Manager에 전달
    def return_leader_condition(self, my_id, exp_name):
        print(f"[Leader] Send Leader's condition to the manager ({manager_ip}:{manager_port})")
        with grpc.insecure_channel(f'{manager_ip}:{manager_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.ManagerStub(channel)
            stub.ReturnCondition(fl_pb2.ConditionMsg(id=my_id, exp_name=exp_name, condition=True))

    # ===== RPC client =====
    def request_mec_model(self, _aggregator, fl_round, finish_condition):
        with grpc.insecure_channel(f"{_aggregator['ip']}:{_aggregator['port']}",
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.AggregatorStub(channel)
            model_weights = self.global_model.get_weights()
            model_layers, model_layers_shape = helper.serialize_model_weights(model_weights)
            address = self.get_pb2_address()
            return_msg = stub.RequestMecModel(fl_pb2.ModelMsg(sender_addr=address, fl_round=fl_round,
                                                              model_layers=model_layers,
                                                              model_layers_shape=model_layers_shape,
                                                              finish = finish_condition))
            accepted = return_msg.accepted
            sender_addr = return_msg.sender_addr
            fl_round = return_msg.fl_round

            request_id = helper.make_request_id(sender_addr.id, fl_round)
            print(
                f"[Leader] Request a mec model to aggregator. "
                f"FL round: {fl_round}, Accepted: {accepted} Aggregator ID: {sender_addr.id}, Request ID: {request_id}")
            if accepted:
                self.lock_aggregator_waiting_queue.acquire()
                self.aggregator_waiting_queue.add(request_id)
                self.lock_aggregator_waiting_queue.release()

    #TODO Optimizer에 learning rate가 안들어가는 오류 해결(현재, optimizer의 default lr(0.002) 사용)
    def init_global_model(self):
        self.global_model = copy.deepcopy(self.base_model)
        self.global_model.compile(optimizer=self.optimizer,
                                  loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                                  metrics=[metrics.SparseCategoricalAccuracy()])
        self.global_model.compute_output_shape(input_shape=(None, 32, 32, 3))

    # Aggregator가 보낸 모델(weight)을 Global Model에 적용
    def aggregate_models(self, fl_round):
        print(f"[Leader] [{fl_round} Round] Aggregates {len(self.registered_aggregators)} Aggregators' MEC models")
        lws = []
        mec_index = []
        max_increase = -float('inf')
        max_index = None
        order=0
        # 현재 리더에 등록된 aggregator 중 모델(weight)을 가졌는지 확인
        for k, _aggregator in self.registered_aggregators.items():
            if f"{fl_round}" not in _aggregator:
                print(f"[Leader] [{fl_round} Round] aggregator {_aggregator['id']}'s MEC model does not exist!")
                pass
            lws.append(_aggregator[f"{fl_round}"]) #lws 리스트에 해당 라운드의 파라미터를 append한다
            mec_index.append(order)
            print(f'[테스트] : {mec_index}')
        # 하이브리드 Aggregation Method를 사용하면,
        # 입력한 aggregation method의 적용 결과를 확인해 가장 좋은 성능을 보인 Method를 적용한다.
        if self.hybrid:
            for i, method in enumerate(self.agg_method):
                agg_ratio = helper.calc_avg_ratio(self.test_data, self.global_model, method, lws, mec_index, self.num_aggs_data)
                sub_w = helper.average_model(lws, agg_ratio)

                sub_model = copy.deepcopy(self.global_model)
                sub_model.set_weights(sub_w)
                sub_results = sub_model.evaluate(x_test, y_test, batch_size=self.eval_batch, verbose=0)
                increase = sub_results[1] - self.acc
                print(f'[Simulator] Aggregation Method : {method}, Previous Acc : {self.acc}, Test Acc : {sub_results[1]}, Increase : {increase}')

                if increase > max_increase:
                    max_increase = increase
                    max_index = i
                    agg_ratio = copy.deepcopy(sub_w)
                del sub_model
                sub_w.clear()
        # 하이브리드 Aggregation Method를 사용하지 않으면,
        # 입력한 aggregation method(첫번째 Method)의 가중치를 적용한다.
        else:
            agg_ratio = helper.calc_avg_ratio(self.test_data, self.global_model, self.agg_method, lws, mec_index, self.num_aggs_data)
            ratio_weight = [self.add_agg_param1, self.add_agg_param2]
            for k, method in enumerate([self.add_agg_method1, self.add_agg_method2]):
                if method != None:
                    ratio_avg = helper.calc_avg_ratio(self.test_data, self.global_model, method, lws, mec_index, self.num_aggs_data)
                    ratio_avg = np.array(ratio_avg) * ratio_weight[k] / 100
                    agg_ratio = np.vstack((agg_ratio, ratio_avg)) 
                    agg_ratio = np.sum(agg_ratio, axis=0)
        print(f"[Leader] [{fl_round} Round] Aggregate by {self.agg_method} Method")

        gw = helper.average_model(lws, agg_ratio)
        self.global_model.set_weights(gw)
        lws.clear()
        #agg_ratio.clear()

    # aggregation method를 특정 상황에서 적용하는 방법
    def change_agg_method(self, module, round, acc):
        if self.adaptive_agg_method.lower() == "round":
            if int(self.adaptive_parameter) > round:
                module.agg_method = 'fedavg'
            else:
                module.agg_method = self.basic_method
        elif self.adaptive_agg_method.lower() == "acc":
            if float(self.adaptive_parameter)/100.0 > acc:
                module.agg_method = 'fedavg'
            else:
                module.agg_method = self.basic_method


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Federated Learning Leader')
    parser.add_argument('-i', '--my_id', dest='my_id', action='store', type=str)
    parser.add_argument('-p', '--my_port', dest='my_port', action='store', type=str, default='50051')    
    parser.add_argument('-m', '--manager_ip', dest='manager_ip', action='store',
                        type=str, required=True, help='<manager ip>:<port>')
    args = parser.parse_args()

    ipfs = FL_IPFS()
    
    #Leader 정보
    my_ip = socket.gethostbyname(socket.gethostname())
    my_port = args.my_port
    if args.my_id is None:
        my_id = helper.make_id('GlobalAggregator', my_ip, my_port)
    else:   
        my_id = args.my_id
    
    # Manager 정보
    manager_ip = args.manager_ip
    manager_port = helper.MANAGER_PORT
    if manager_ip.find(':') > 0:
        manager_port = manager_ip.split(':')[1]
        manager_ip = manager_ip.split(':')[0]        
        
    print(f"#HFL#[GlobalAggregator] (id={my_id}, ip={my_ip}:{my_port}, manager={manager_ip}:{manager_port})", flush=True)

    # GPU Dynamic allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    logging.basicConfig()
    flLeader = None

    while True:
        # Initialize Leader
        if flLeader is None:
            flLeader = FlLeader()
        
        flLeader.set_address(my_id, my_ip, my_port)
        flLeader.rpc_serve()
        flLeader.register_Leader_to_manager()
        flLeader.check_config_file()
        print('\n[Leader] We get the Config Datas!\n')
        flLeader.init_global_model()
        
        if helper.WANDB:
            wandb_ID = helper.WANDB_ID
            wandb_PW = helper.WANDB_API
            if wandb_PW != None:
                wandb.login(key=f"{wandb_PW}")
            
            wandb.init(project="ETRI_exp", mode='online', group=flLeader.group_name, entity=f'{wandb_ID}', name=f"{flLeader.exp_name}_simul_'{flLeader.simulate}'_agg_{flLeader.num_target_collecting}_user_{flLeader.num_user}")
            wandb.config.update({'exp' : flLeader.exp_name, 'simulation' : flLeader.simulate, 'n_agg' : flLeader.num_target_collecting, 'n_user' : flLeader.num_user,
                                'model' : flLeader.model, 'optimizer' : flLeader.optimizer, 'dataset' : flLeader.dataset, 'fraction' : flLeader.fraction, 'total_aggregation_method' : flLeader.total_agg_method, 'aggregation_method' : flLeader.agg_method, 
                                'additional_method1' : flLeader.add_agg_method1, 'additional_method2' : flLeader.add_agg_method2, 'adaptive_agg_method' : flLeader.adaptive_agg_method, 
                                'adaptive_parameter' : flLeader.adaptive_parameter})
        # get Test dataset
        x_test, y_test = flLeader.test_data
        y_test = np.argmax(y_test, axis=1)

        # Leader Information
        data_num_test = len(y_test)
        num_agg = flLeader.num_init_aggregators

        # Federated Learning
        FL_INIT_TIME = 2  # seconds
        global_losses = []
        global_accs = []
        global_fl_start_time = datetime.now()
        dict_df = {'Round': [], 'TestAcc': []}

        print(f"[Leader] ====== Model: {flLeader.model} (#Dataset: {flLeader.dataset} (#test: {data_num_test})")
        print(f"[Leader] ====== #MECs: {num_agg}")
        print(f"[Leader] ====== #Total MEC rounds: {flLeader.global_round}")
        print(f"[Leader] ====== #Optimizer: {flLeader.optimizer}, #Evaluation batch size: {flLeader.eval_batch}")
        
        #Using Simulator
        if flLeader.simulate:
            # HFL Infrastructure Configuration
            print("[Leader] ====== Start Federated Learning simulator\n")
            print("Init Models")
            t1 = datetime.now()
            EDGES = flLeader.compose_user(flLeader.x_dataset, flLeader.y_dataset)
            MECS = flLeader.compose_mec(EDGES)
            SERVERS = flLeader.compose_server(MECS)
            t2 = datetime.now()
            init_time = t2 - t1
            print(f'init model Time(sec): {round(init_time.total_seconds(), 2)}')

            ROUND = flLeader.global_round
            eval_every = flLeader.eval_every
            dict_df = {'Round': [], 'TestAcc': []}

            # learning rate decay on/off            
            USE_LR_DECAY = False
            if flLeader.lr_decay:
                lr = flLeader.learning_rate
                USE_LR_DECAY=True

            # aggregation method switching logic on(acc,round)/off(no_adapt)
            if flLeader.adaptive_agg_method != 'no_adapt':
                print("Adaptive Aggregation, starting with FedAvg, After,", flLeader.adaptive_agg_method, flLeader.adaptive_parameter, ", ", flLeader.agg_method)
            
            # HFL Training
            for r in range(1, ROUND+1):
                SERVERS.train()
                
                # Learning rate decay logic
                if USE_LR_DECAY and flLeader.lr_decay and not((r+1) % flLeader.lr_decay_round):
                    lr *= flLeader.lr_decay
                    SERVERS.set_lr(lr)

                # Test Acc 산출을 조정하는 파트
                if ((r-1) % eval_every == 0) or (r >= ROUND - 20):
                    results = SERVERS.test()
                    print(f'Round: {r}, Loss: {results[0]}, Acc: {results[1]:.2%}')
                    SERVERS.acc = results[1]
                    dict_df['Round'].append(r)
                    dict_df['TestAcc'].append(round(results[1]*100, 2))

                    # Wandb Upload
                    # TODO Wandb -> TensorBoard로 업로드 전환
                    if helper.WANDB:
                        wandb.log({
                            "Test Acc": round(results[1]*100, 2),
                            "Test Loss": round(results[0], 2),
                        }, step=r)

                #집계 방법 전환 로직 선정(에폭, Test Acc)
                flLeader.change_agg_method(SERVERS, r, results[1])
                #if (r - 1) % 100 == 0:
                    #os.makedirs('./checkpoints', exist_ok=True)
                    #SERVER.model.save(f'checkpoints/R:{r}_{args.exp_name}')
            global_fl_end_time = datetime.now()
            global_fl_elapsed_time = global_fl_end_time - global_fl_start_time

        else:
            print("[Leader] ====== Start MEC Federated Learning\n")
            # Register Aggregators
            while len(flLeader.registered_aggregators) < flLeader.num_init_aggregators:
                print(f"[Leader] Waiting for aggregators... Current registered: " f"{len(flLeader.registered_aggregators)} / {flLeader.num_init_aggregators}", flush=True)
                time.sleep(1)
            
            print(f"[Leader] {len(flLeader.registered_aggregators)} users are gathered! \n ")
            for idx, (_, aggregator) in enumerate(flLeader.registered_aggregators.items()):
                print(f"[{idx + 1}] {aggregator}")
            time.sleep(FL_INIT_TIME)
            print(f"[Leader] Let's start FL ", flush=True)
            
            for r in range(1, flLeader.global_round + 1):
                global_round_start_time = datetime.now()
                print(f"[Leader] [{r}/{flLeader.global_round} ROUND] ========================== Start Global {r} ROUND ==========================")

                # 1. [FL] Request MEC models to each aggregator
                for idx, (_, aggregator) in enumerate(flLeader.registered_aggregators.items()):
                    print(f"[Leader] [{r}/{flLeader.global_round} ROUND] ====== Request MEC model to {len(flLeader.registered_aggregators)} Aggregators", flush=True)
                    flLeader.request_mec_model(aggregator, r, False)
                time.sleep(FL_INIT_TIME)

                # 2. [FL] Wait for the MEC models uploaded from the aggregators
                while True:
                    flLeader.lock_aggregator_waiting_queue.acquire()
                    remaining_mec_models = len(flLeader.aggregator_waiting_queue)
                    if remaining_mec_models == 0:
                        flLeader.lock_aggregator_waiting_queue.release()
                        break

                    print(f"[Leader] [{r}/{flLeader.global_round} ROUND] Waiting MEC models... ({len(flLeader.collected_aggregators)}/{flLeader.num_target_collecting})", flush=True)
                    flLeader.lock_aggregator_waiting_queue.release()
                    time.sleep(1)

                flLeader.aggregate_models(r)    

                # 3. [FL] Evaluation the global model
                results = flLeader.global_model.evaluate(x_test, y_test, batch_size=64, verbose=0)
                print(f"[Leader] [{r}/{flLeader.global_round} ROUND] ====== Global model Evaluation Accuracy: {round(results[1]*100, 2)}%", flush=True)
                global_losses.append(results[0])
                global_accs.append(results[1])

                # 4. [FL] Aggregation the MEC models
                #집계 방법 변화(에폭, Test ACC)
                flLeader.change_agg_method(flLeader, r, results[1])
                print(f"[Leader] [{r}/{flLeader.global_round} ROUND] ====== Global model Aggregation using (Collected aggregators: {len(flLeader.collected_aggregators)}, Aggregation Method : {flLeader.agg_method})", flush=True)


                # 5. [FL] Write the results to tensorboard
                '''with tf.summary.create_file_writer(tensorboard_log_dir).as_default():
                    tf.summary.scalar('loss', results[0], step=r)
                    tf.summary.scalar('accuracy', results[1], step=r)'''
                
                # 6. [Wandb Upload]
                
                dict_df['Round'].append(r)
                dict_df['TestAcc'].append(round(results[1]*100, 2))

                if helper.WANDB:
                    wandb.log({
                        "Test Acc": round(results[1]*100, 2),
                        "Test Loss": round(results[0], 2),
                    }, step=r)
                time.sleep(FL_INIT_TIME)

                # Calculate the elapsed time
                global_round_end_time = datetime.now()
                global_round_elapsed_time = global_round_end_time - global_round_start_time
                print(
                    f"[Leader] [{r}/{flLeader.global_round} round] ====== Finish MEC round (round elapsed time: {round(global_round_elapsed_time.total_seconds(), 2)} seconds)",
                    flush=True)
            
            # 5. [FL] Evaluation the final global model
            for idx, (_, aggregator) in enumerate(flLeader.registered_aggregators.items()):
                flLeader.request_mec_model(aggregator, r, True)
            results = flLeader.global_model.evaluate(x_test, y_test, batch_size=flLeader.eval_batch, verbose=1)

            global_fl_end_time = datetime.now()
            global_fl_elapsed_time = global_fl_end_time - global_fl_start_time

            max_acc = max(global_accs)
            max_round = global_accs.index(max_acc)
            max_acc_loss = global_losses[max_round]

            try:
                target_index = next(i for i, acc in enumerate(global_accs) if acc >= 97)
                target_acc = global_accs[target_index]
            except StopIteration:
                target_index = 'None'
                target_acc = f'Could not reach to {flLeader.stop_acc}%'
            
            print(f"[Leader] [{r} ROUND] ====== Finish MEC Federated Learning (MEC FL elapsed time: {round(global_fl_elapsed_time.total_seconds(), 2)} seconds)")
            print(f"[Leader] [{r} ROUND] ====== Final Accuracy: {round(results[1] * 100, 2)}%, Loss: {results[1]}")
            print(f"[Leader] [{r} ROUND] ====== Maximum Accuracy: {max_round} round (Accuracy: {round(max_acc * 100, 2)}%, Loss: {max_acc_loss})")
            print(f"[Leader] [{r} ROUND] ====== First Reach round to target accuracy ({flLeader.stop_acc}%): {target_index} round (Accuracy: {target_acc}%)")
            print(f"[Leader] [{r} ROUND] ====== Number of model sending to aggregators: {flLeader.global_round * num_agg} (round x aggregators)")
            print(f"[Leader] [{r} ROUND] ====== Total size of sent models to aggregators: {flLeader.global_round * num_agg * 6.6}MB (round x aggregators x model_size)")
            print(f"[Leader] [{r} ROUND] ====== Number of model receiving to aggregators: {flLeader.global_round * num_agg} (round x aggregators)")
            print(f"[Leader] [{r} ROUND] ====== Total size of received models to aggregators: {flLeader.global_round * num_agg * 6.6}MB (round x aggregators x model_size)")
            print(f"[Leader] [{r} ROUND] ====== Number of aggregators' participation: {num_agg}")

        if helper.WANDB:
            wandb.init()
            wandb.finish()

        flLeader.send_final_results_to_manager(flLeader.global_round,results[1],results[0],num_agg,global_fl_elapsed_time.total_seconds())
        flLeader.__init__()
        flLeader.return_leader_condition(my_id, flLeader.exp_name)
        '''df = pd.DataFrame(dict_df)
        os.makedirs('./csv_results', exist_ok=True)
        f_name = f'Exp_{datetime.now()}.csv'

        df.to_csv(f'./csv_results/{f_name}')

        wandb.save(f'./csv_results/{f_name}')'''
        