import os
import argparse
import copy
import logging
import random
import socket
import time
import threading
from concurrent import futures
from datetime import datetime

import google
import grpc
import tensorflow as tf
from tensorflow.python.keras import models, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import *
from dataclasses import dataclass

import fl_pb2
import fl_pb2_grpc

import Library.helper as helper
from Library.datasets import *
from Library.models import *
from Library.ipfs import FL_IPFS

import FL_Manager

MAX_MESSAGE_LENGTH = helper.MAX_MESSAGE_LENGTH

# tensorboard
tensorboard_log_dir = "tb_logs"

# class AggregatorRpcService
# Aggregation RPC 함수 정의
# -> Aggregation은 RPC Server를 실행하여 외부에 RPC 함수를 제공
# -> Aggregator는 해당 RPC 함수를 통해 Leader 및 Userset과 통신하여 연합학습을 수행
class AggregatorRpcService(fl_pb2_grpc.Aggregator):
    def __init__(self, aggregator):
        super().__init__()
        self.aggregator: FlAggregator = aggregator

    # Request~~ : 상위 계층에서 받아오는 함수
    # -> Global FL round 시작용 함수
    # -> Leader는 Global FL의 매 라운드마다 해당 함수를 호출하여 Aggregator에게 MEC Model을 요청함
    # -> 계층형 연합학습 (Hierarchical FL==True) 인 경우에만 수행
    def RequestMecModel(self, model_msg, context):
        print(f"[Aggregator] [{model_msg.fl_round} Round] MEC model request by the leader:", context.peer(), flush=True)

        my_addr = self.aggregator.get_pb2_address()
        fl_round = model_msg.fl_round
        finish_condition = model_msg.finish
        self.aggregator.finish_condition = finish_condition
        if self.aggregator.running_MEC_FL == False and finish_condition == False:
            self.aggregator.apply_to_mec_model(model_msg)
            self.aggregator.running_MEC_FL = True
            self.aggregator.current_global_round = model_msg.fl_round
            return fl_pb2.ReturnMsg(accepted=True, sender_addr=my_addr, fl_round=model_msg.fl_round, num_trained_data=model_msg.num_trained_data)
        elif finish_condition == True:
            self.aggregator.leader_online_condition = False
        print(f"[Aggregator] [{fl_round} Round] Reject RequestMecModel: ")
        return fl_pb2.ReturnMsg(accepted=False, sender_addr=my_addr, fl_round=model_msg.fl_round, num_trained_data=model_msg.num_trained_data)

    # RegisterUserSet()
    # -> userset 등록 함수
    # -> userset은 해당 함수를 통해 자신의 정보를 Aggregator에게 등록함
    def RegisterUserSet(self, addr_set_msg, context):
        for idx, addr_msg in enumerate(addr_set_msg.address_set):
            self.aggregator.register_user(addr_msg)

        my_addr = self.aggregator.get_pb2_address()

        return fl_pb2.ReturnMsg(accepted=True, sender_addr=my_addr)

    # ReturnUserModelSet()
    # -> User Model 등록 함수
    # -> user는 자신이 생성(학습한) 결과인 user model을 해당 함수를 통해 Aggregator에게 전달
    def ReturnUserModelSet(self, model_set_msg, context):
        # print(f"[Aggregator] {len(model_set_msg.model_set)} Users' model received")
        self.aggregator.lock_user_waiting_queue.acquire()
        fl_round = 0
        for idx, model_msg in enumerate(model_set_msg.model_set):
            fl_round = model_msg.fl_round
            user_id = model_msg.sender_addr.id
            num_trained_data = model_msg.num_trained_data
            request_id = helper.make_request_id(user_id, fl_round)

            # MEC 내 User가 가지고 있는 각각의 데이터 개수
            if len(self.aggregator.num_userset_data) < self.aggregator.num_target_collecting:
                self.aggregator.num_userset_data.append(num_trained_data)
                self.aggregator.num_aggregator_data += num_trained_data

            # ipfs == True 、 download model weight 
            if flAggregator.ipfs == True:
                model_hash = model_msg.model_hash
                print(f"[IPFS] Recived local weight file hash : {model_hash}")
                downloaded_model_file_name = ipfs.download_from_ipfs(model_hash)
            
                request_id_exist = request_id in self.aggregator.user_waiting_queue
                if request_id_exist:
                    self.aggregator.register_user_model_IPFS(model_msg,downloaded_model_file_name)
                    self.aggregator.user_waiting_queue.remove(request_id)
                    os.remove(downloaded_model_file_name)
                    #os.remove(downloaded_model_file_name) <- ipfs 파일 저장후 자동삭제하는 code.
            else:
                request_id_exist = request_id in self.aggregator.user_waiting_queue
                if request_id_exist:
                    self.aggregator.register_user_model(model_msg)
                    self.aggregator.user_waiting_queue.remove(request_id)

        print(f"[Userset 모델셋 합산 확인] 데이터 수 총합 : {self.aggregator.num_aggregator_data}")
        print(f"[Userset 모델셋 입력확인] 한 유저셋 데이터 수: {self.aggregator.num_userset_data}")
        self.aggregator.lock_user_waiting_queue.release()
        return fl_pb2.ReturnMsg(fl_round=fl_round, accepted=True)

    def ConfigSending(self, request, context):
        config_dict = helper.convert_proto_to_dict(request)

        FL_Service = config_dict['FL_Service']
        # ====== Participate_Info ======
        self.aggregator.total_user = int(FL_Service['user'])
        self.aggregator.is_hfl = helper.str2bool(FL_Service['is_hfl'])
        self.aggregator.top_weight_save_percent = helper.int_default(FL_Service,'top_weight_save_percent', 100)
        self.aggregator.learning_layer_mode = helper.bool_default(FL_Service, 'learning_layer_mode')
        self.aggregator.ipfs = helper.bool_default(FL_Service, 'ipfs')
        end_condition = helper.list_default(FL_Service, 'end_condition', '5, 99')
        self.aggregator.global_round = int(end_condition[0])
        self.aggregator.stop_acc = float(end_condition[1])

        # ====== Network_Info ======
        My_network = config_dict['network']
        self.aggregator.my_id = My_network['id']
        self.aggregator.mec_index = int(My_network['index'])

        if self.aggregator.is_hfl:
            Network_data = config_dict['send_network']
            self.aggregator.leader_ip = Network_data['ip']
            self.aggregator.leader_port = Network_data['port']

        # ====== Model/Data_Info ======
        self.aggregator.model = helper.str_default(FL_Service, 'model', 'mcmahan2NN')
        self.aggregator.pretrained = helper.bool_default(FL_Service, 'pretrained')
        self.aggregator.optimizer = helper.str_default(FL_Service, 'optimizer', 'adam')
        self.aggregator.base_model = define_model(self.aggregator.model , self.aggregator.pretrained)
        self.aggregator.dataset = helper.str_default(FL_Service, 'dataset', 'mnist')
        self.aggregator.train_data, self.aggregator.test_data = load_dataset(self.aggregator.dataset)
        self.aggregator.fraction = helper.int_default(FL_Service, 'fraction', 100)
        self.aggregator.agg_round = helper.int_default(FL_Service, 'aggregator_round', 3)
        self.aggregator.eval_batch = helper.int_default(FL_Service, 'evaluation_batch_size', 64)
        self.aggregator.num_init_users = int(FL_Service['user_per_agg'])
        self.aggregator.num_target_collecting = int(FL_Service['user_per_agg'])
        
        return google.protobuf.empty_pb2.Empty()

# group_users_per_userset()
# -> User selection (based on fraction)을 통해 랜덤하게 선택된 유저들을 같은 Userset에 속하는 유저끼리 그룹핑
# -> 그 결과로, userset에 대한 list (usersets)를 반환
# -> 실제 프로세스는 userset 단위로 실행되기 있기 때문에, 실제 학습 요청 시 userset 단위를 사용함
def group_users_per_userset(selected_users):
    # if users' ip and port are the same, these users are under the same userset process
    usersets = {}
    for user in selected_users:
        userset_key = f"{user['ip']}:{user['port']}"
        if userset_key not in usersets:
            userset_value = []
            usersets[userset_key] = userset_value
        usersets[userset_key].append(user)
    return usersets


class FlAggregator:
    def __init__(self):
        self.is_hfl = None
        self.address = {}
        self.current_global_round = 0
        self.mec_model = None
        self.lock_user_waiting_queue = threading.Lock()
        self.user_waiting_queue = set()
        self.grpc_server = None
        self.connected_leader = None
        self.registered_users = {}  # connected users
        self.selected_users = []  # sampled users to request local training
        self.collected_users = []  # users who have sent user model in current round
        self.num_userset_data = []
        self.num_aggregator_data = 0
        self.running_MEC_FL = False
        self.finish_condition = False
        self.mec_index = None
        self.leader_ip = None
        self.leader_port = None
        self.model = None
        self.optimizer = None
        self.pretrained = None
        self.dataset = None
        self.fraction = None
        self.global_round = None
        self.base_model = None
        self.test_data = None
        self.agg_round = None
        self.eval_batch = None
        self.num_init_users = None
        self.num_target_collecting = None
        self.is_hfl = None
        self.ipfs = None
        self.learning_layer_mode = None
        self.top_weight_save_percent = None
        self.leader_online_condition = False
        self.agg_online_condition = False

    def check_config_file(self):
        while self.num_target_collecting is None:
            print("[Aggregator] Waiting for input parameter to be present by Client...")
            time.sleep(5)
    '''
    def check_online_condition(self):
        while self.agg_online_condition == False or self.leader_online_condition == False:
            print("[Aggregator] Waiting for previous FL...")
            time.sleep(5)'''
    
    #안쓰는 함수
    '''def set_fl_parameters(self, num_mecs, mec_index, num_total_users, is_iid):
        self.num_mecs = num_mecs
        self.mec_index = mec_index
        self.num_total_users = num_total_users
        self.is_iid = is_iid'''

    def set_address(self, _id, _ip, _port):
        self.address['id'] = f"{_id}"
        self.address['ip'] = f"{_ip}"
        self.address['port'] = f"{_port}"

    def get_pb2_address(self):
        _id = self.address['id']
        _ip = self.address['ip']
        _port = self.address['port']
        # print(self.address['id'], self.address['ip'], self.address['port'])

        return fl_pb2.AddressMsg(id=_id, ip=_ip, port=_port)

    def init_mec_model(self):
        self.mec_model = copy.deepcopy(self.base_model)
        self.mec_model.compile(optimizer=self.optimizer,
                               loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                               metrics=[metrics.SparseCategoricalAccuracy()])
        self.mec_model.compute_output_shape(input_shape=(None, 32, 32, 3))

    # Global FL를 위한 학습 전 이번 Round에서 Leader에게 받은 Global Model을 적용하는 함수
    def apply_to_mec_model(self, _model_msg):
        layers = _model_msg.model_layers
        layers_shape = _model_msg.model_layers_shape
        model_weights = helper.deserialize_model_weights(layers, layers_shape)
        #model_weights = _model_msg.model_weights
        self.mec_model.set_weights(model_weights)

    # rpc server 실행 함수
    def rpc_serve(self):
        my_port = self.address['port']
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                             options=[
                                 ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                 ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                 ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                             ])
        rpcService = AggregatorRpcService(self)
        fl_pb2_grpc.add_AggregatorServicer_to_server(rpcService, server)
        server.add_insecure_port(f'[::]:{my_port}')

        server.start()
        print(f"[Aggregator] Rpc service started!, listening on {my_port}")

        self.grpc_server = server
        # server.wait_for_termination()
    
    # ===== Manager's RPC 함수 호출 / Manager 에게 자신을 등록하는 함수
    def register_agg_to_manager(self):
        print(f"[Aggregator] Send aggregator registration request to the manager ({manager_ip}:{manager_port})")
        with grpc.insecure_channel(f'{manager_ip}:{manager_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.ManagerStub(channel)
            my_addr = self.get_pb2_address()
            ret_msg = stub.RegisterAggregator(my_addr)

            print(f"[Aggregator] Registration result: {ret_msg.accepted}")
    
    def return_agg_condition(self, my_id):
        print(f"[Aggregator] Send aggregator's condition to the manager ({manager_ip}:{manager_port})")
        with grpc.insecure_channel(f'{manager_ip}:{manager_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.ManagerStub(channel)
            stub.ReturnCondition(fl_pb2.ConditionMsg(id=my_id, condition=True))

    # ===== Leader's RPC 함수 호출 =====
    # Ledaer에게 결과 모델인 MEC Model을 return하는 함수 (Hierarchical FL==True 인 경우에만 수행)
    def return_mec_model_hash(self, fl_round,model_hash):
        with grpc.insecure_channel(f'{self.leader_ip}:{self.leader_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.LeaderStub(channel)
            my_addr = self.get_pb2_address()
            ret_msg = stub.ReturnMecModel(fl_pb2.ModelMsg(fl_round=fl_round, sender_addr=my_addr,
                                                          num_trained_data= self.num_aggregator_data,
                                                          mec_index= self.mec_index,
                                                          model_hash=model_hash))
            print(f"[Aggregator] [{ret_msg.fl_round} Round] Upload my mec model to the leader."
                  f"Result: {ret_msg.accepted}")
            
    def return_mec_model(self, fl_round, model_layers, model_layers_shape):
        with grpc.insecure_channel(f'{self.leader_ip}:{self.leader_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.LeaderStub(channel)
            my_addr = self.get_pb2_address()
            ret_msg = stub.ReturnMecModel(fl_pb2.ModelMsg(fl_round=fl_round, sender_addr=my_addr,
                                                          model_layers=model_layers,
                                                          model_layers_shape=model_layers_shape,
                                                          num_trained_data= self.num_aggregator_data,
                                                          mec_index= self.mec_index))
            print(f"[Aggregator] [{ret_msg.fl_round} Round] Upload my mec model to the leader."
                  f"Result: {ret_msg.accepted}")


    # ===== Userset's RPC 함수 호출 =====
    # Userset에 속한 user들에 대한 정보 습득
    def get_user_set_info(self, ip_port):
        with grpc.insecure_channel(f"{ip_port}",
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.UserSetStub(channel)
            user_set_info_msg = stub.GetUserSetInfo(
                fl_pb2.BasicMsg(none=True)
            )

            return user_set_info_msg

    # ===== Userset's RPC 함수 호출 =====
    # Userset에게 User model을 요청
    # MEC FL의 매 라운드마다 호출 됨
    # 해당 요청을 받은 User는 학습을 진행한 후 ReturnUserModelSet()을 통해 User model을 return
    def request_user_model_set(self, ip_port, userset, fl_round, finish_condition):
        with grpc.insecure_channel(f"{ip_port}",
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.UserSetStub(channel)
            #model_weights.append(copy.deepcopy(self.mec_model.get_weights()))
            
            model_weights = self.mec_model.get_weights()
            model_layers, model_layers_shape = helper.serialize_model_weights(model_weights)
            my_address = self.get_pb2_address()
            #mec_mapping_data = {'id': self.mec_index, 'num': self.num_aggregator_data}
            my_model = fl_pb2.ModelMsg(sender_addr=my_address, fl_round=fl_round,
                                       model_layers=model_layers, model_layers_shape=model_layers_shape, 
                                       num_trained_data=self.num_aggregator_data, mec_index = self.mec_index, finish= finish_condition)

            target_users = []
            for user in userset:
                target_users.append(fl_pb2.AddressMsg(id=user['id'], ip=user['ip'], port=user['port']))

            return_set_msg = stub.RequestUserModelSet(
                fl_pb2.ReqUserModelSetMsg(model=my_model, address_set=target_users))
            print(f"[Aggregator] Request user model to userset {ip_port} for {len(userset)} users")
            self.lock_user_waiting_queue.acquire()
            for return_msg in return_set_msg.return_set:
                accepted = return_msg.accepted
                sender_addr = return_msg.sender_addr
                fl_round = return_msg.fl_round
                request_id = helper.make_request_id(sender_addr.id, fl_round)
                # print(
                #     f"[Aggregator] Request user model to userset {ip_port} for {len(userset)} users"
                #     f"FL round: {fl_round}, Accepted: {accepted} User ID: {sender_addr.id}, Request ID: {request_id}")
                if accepted:
                    self.user_waiting_queue.add(request_id)
            self.lock_user_waiting_queue.release()

    def register_leader(self, message):
        leader = {'id': message.id, 'ip': message.ip, 'port': message.port}
        self.connected_leader = leader

    def register_user(self, message):
        new_user = {'id': message.id, 'ip': message.ip, 'port': message.port}
        self.registered_users[message.id] = new_user

    # MEC FL 라운드마다 각 user가 전달한 user model을 "collected_users" list에 추가
    # flow -> local(user) -> ipfs -> aggregator
    def register_user_model_IPFS(self, message,model_weight_IPFS):
        model_weights_IPFS = FL_IPFS.load_model_weights_from_ipfs(model_weight_IPFS)
        fl_round = message.fl_round
        user_id = message.sender_addr.id
        _user = self.registered_users[user_id]
        _user[f"{fl_round}"] = model_weights_IPFS
        self.collected_users.append(_user)

    def register_user_model(self, message):
        model_weights = helper.deserialize_model_weights(message.model_layers, message.model_layers_shape)
        fl_round = message.fl_round
        user_id = message.sender_addr.id
        _user = self.registered_users[user_id]
        _user[f"{fl_round}"] = model_weights 
        self.collected_users.append(_user)


    # MEC FL 라운드마다 user들로부터 전달받은 user model들을 aggregation하여 MEC model을 생성
    def aggregate_models(self, fl_round):
        print(f"[Aggregator] [{fl_round} Round] Aggregates {len(self.collected_users)} users' models", flush=True)
        lws = [] # 가중치 저장
        for _user in self.collected_users:
            if f"{fl_round}" not in _user:
                print(f"[Aggregator] [{fl_round} Round] user {_user['id']}'s MEC model does not exist!")
            lws.append(_user[f"{fl_round}"])

        # Aggregate all the local models
        if flAggregator.learning_layer_mode == True:
            gw = helper.average_model(lws, 'Layer')
        else:
            gw = helper.average_model(lws, None)
        self.mec_model.set_weights(gw)


    # 등록된 user 중 fraction만큼 user selection 수행
    def sample_users(self):
        num_selected = max(1, int((self.fraction/100) * len(self.registered_users)))
        selected_user_keys = random.sample(list(self.registered_users), num_selected)
        selected_user_keys = sorted(selected_user_keys)
        # print(f"[Aggregator] [{_fl_round} Round] Selected users: {selected_user_keys}")
        for user_key in selected_user_keys:
            self.selected_users.append(self.registered_users[user_key])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Federated Learning Aggregator')
    parser.add_argument('-i', '--my_id', dest='my_id', action='store', type=str)
    parser.add_argument('-p', '--my_port', dest='my_port', action='store', type=str, default='50081')
    parser.add_argument('-m', '--manager_ip', dest='manager_ip', action='store',
                        type=str, required=True, help='<manager ip>:<port>')    
    args = parser.parse_args()
    ipfs = FL_IPFS()
    
    #Aggregator 통신 정보
    my_ip = socket.gethostbyname(socket.gethostname())
    #TODO IP 지정 필요
    my_port = args.my_port
    if args.my_id is None:
        my_id = 'agg_' + my_ip + my_port
    else:   
        my_id = args.my_id
    
    # Manager 정보
    manager_ip = args.manager_ip
    manager_port = helper.MANAGER_PORT
    if manager_ip.find(':') > 0:
        manager_port = manager_ip.split(':')[1]
        manager_ip = manager_ip.split(':')[0]        
    
    print(f"#HFL#[Aggregator] (id={my_id}, ip={my_ip}:{my_port}, manager={manager_ip}:{manager_port})", flush=True)
    
    flAggregator = None
    logging.basicConfig()

    # GPU Dynamic allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    while True:
        # Initialize Aggregator
        if flAggregator is None:
            flAggregator = FlAggregator()

        flAggregator.set_address(my_id, my_ip, my_port)
        flAggregator.rpc_serve()

        # Manager에게 Leader 정보 등록
        flAggregator.register_agg_to_manager()
        flAggregator.check_config_file()
        print('\n[Aggregator] We get the Config Datas!\n')
        #time.sleep(10)
        flAggregator.init_mec_model()

        if flAggregator.is_hfl and flAggregator.my_id == "aggregator":
            print(f"[Aggregator] When HFL, you must input an unique aggregator's id \n ")
            exit()
        
        # Register Users
        while len(flAggregator.registered_users) < flAggregator.num_init_users:
            print(
                f"[Aggregator] Waiting for users... Current registered: "
                f"{len(flAggregator.registered_users)} / {flAggregator.num_init_users}", flush=True)
            time.sleep(2)
            #time.sleep(10)

        fl_init_time = 2  # seconds
        print(f"[Aggregator] {len(flAggregator.registered_users)} users are gathered! \n ")
        # for idx, (_, user) in enumerate(flAggregator.registered_users.items()):
        #     print(f"[{idx + 1}] {user}")
        time.sleep(fl_init_time)

        if flAggregator.is_hfl:
            flmanager = FL_Manager.FlManager()
            flmanager.register_to_leader(flAggregator.leader_ip,flAggregator.leader_port,my_id,my_ip,my_port)

        print(f"[Aggregator] Let's start FL ", flush=True)

        #테스트셋 업로드
        x_test, y_test = flAggregator.test_data
        y_test = np.argmax(y_test, axis=1)
        data_num_test = len(y_test)
        data_num_train = flAggregator.num_aggregator_data
        

        # 모델의 크기 확인
        #flAggregator.base_model.save('model_size.h5')
        model_num_params = flAggregator.base_model.count_params()
        #model_size_bytes = os.path.getsize('model_size.h5')
        #model_size = model_size_bytes / (1024 * 1024)
        num_users_mec = len(flAggregator.registered_users)

        cost_function = "SparseCategoricalCrossEntropy"

        learning_rate_decay_step = None
        learning_rate_decay_rate = None
        regularization = "L2 Regularization"
        weight_initialization = "glorot_uniform"

        # Set the information
        # 실제 연합학습에 사용되는 파라미터는 실제로 설정되며,
        # 후에 추가할 예정인 임시 파라미터는 위에서 설정한 dummy 값으로 설정됨
        flAggregator.model_info = f"{flAggregator.model} (#params: {model_num_params}, size: {'model_size'}"
        flAggregator.data_info = f"{flAggregator.dataset} (#train: {data_num_train}, #test: {data_num_test}"
        flAggregator.batch_size_info = f"evaluation_batch_size: {flAggregator.eval_batch}"
        flAggregator.cost_function = cost_function
        flAggregator.regularization = regularization
        flAggregator.weight_initialization = weight_initialization
        
        # Federated Learning
        while True:
            if flAggregator.finish_condition == True:
                break
            if flAggregator.is_hfl:
                mec_round = flAggregator.agg_round
                R = flAggregator.current_global_round
                print(f"[Aggregator] [{R} ROUND] Waiting for MEC FL request from the leader", flush=True)
                time.sleep(5)
                #time.sleep(10)
            else:
                mec_round = flAggregator.global_round
            
            if flAggregator.running_MEC_FL or (not flAggregator.is_hfl):
                R = flAggregator.current_global_round
                mec_losses = []
                mec_accs = []

                mec_fl_start_time = datetime.now()
                print(f"[Aggregator] [{R} ROUND] ====== Start MEC Federated Learning")
                print(f"[Aggregator] [{R} ROUND] ====== Model: {flAggregator.model} (#params: {model_num_params}, size: {'model_size'}), Data: {flAggregator.dataset} (#train: {data_num_train}, #test: {data_num_test})")
                print(f"[Aggregator] [{R} ROUND] ====== #Users: {num_users_mec}, #Data per user: {data_num_train/num_users_mec}")
                print(f"[Aggregator] [{R} ROUND] ====== #Total MEC rounds: {flAggregator.agg_round}, Selection fraction: {flAggregator.fraction}%, Collection fraction: {flAggregator.fraction}%")
                print(f"[Aggregator] [{R} ROUND] ====== Evaluation batch size: {flAggregator.eval_batch}")
                print(f"[Aggregator] [{R} ROUND] ====== Cost function: {cost_function}, Evaluation batch size: {flAggregator.eval_batch}")
                #print(f"[Aggregator] [{R} ROUND] ====== Optimizer: {optimizer}, earning rate: {flAggregator.local_learning_rate}, Learning rate decay step: {learning_rate_decay_step}, Learning rate decay rate: {learning_rate_decay_rate}")
                print(f"[Aggregator] [{R} ROUND] ====== Regularization: {regularization}, Weight initialization: {weight_initialization}")
                #print(f"[Aggregator] [{R} ROUND] ====== Target accuracy: {flAggregator.target_accuracy}, Early stopping patience: {flAggregator.early_stopping_patience}", flush=True)

                #mec_avg_model =[]
                for r in range(1, mec_round+1):
                    mec_round_start_time = datetime.now()
                    print(f"[Aggregator] [{R} ROUND] [{r}/{mec_round} round] ========================== Start MEC {r} round ==========================")

                    # 1. [FL] Samples users to request local training
                    flAggregator.sample_users()
                    usersets = group_users_per_userset(flAggregator.selected_users)
                    user_id_arr = []
                    for ip_port, userset in usersets.items():
                        for user in userset:
                            user_id_arr.append(user['id'])

                    print(f"[Aggregator] [{R} ROUND] [{r}/{mec_round} round] ====== User Selection (Total Registered users: {len(flAggregator.registered_users)}, Selection Fraction: {flAggregator.fraction}%)")
                    print(f"[Aggregator] [{R} ROUND] [{r}/{mec_round} round] ====== Selected users ({len(flAggregator.selected_users)} users)", flush=True)  # {user_id_arr})")
                    flAggregator.num_target_collecting = len(flAggregator.selected_users)

                    # 2. [FL] Requests user models to each user
                    idx = 0
                    for ip_port, userset in usersets.items():
                        for user in userset:
                            # print(f"[Aggregator] [{R} ROUND] [{r} round] Request user model to user: {user['id']} ({ip_port}) ({idx + 1}/{len(flAggregator.selected_users)})")
                            idx += 1
                        flAggregator.request_user_model_set(ip_port, userset, r, False)
                    time.sleep(fl_init_time)
                    print(f"[Aggregator] [{R} ROUND] [{r}/{mec_round} round] ====== Request user model to {len(flAggregator.selected_users)} users", flush=True)
                    # 2. [FL] Waits for the user models uploaded from the users
                    while True:
                        flAggregator.lock_user_waiting_queue.acquire()
                        remaining_user_models = len(flAggregator.user_waiting_queue)
                        if remaining_user_models == 0 or \
                                len(flAggregator.collected_users) >= flAggregator.num_target_collecting:
                            flAggregator.lock_user_waiting_queue.release()
                            break

                        print(f"[Aggregator] [{R} ROUND] [{r}/{mec_round} round] Waiting user models... ({len(flAggregator.collected_users)}/{flAggregator.num_target_collecting})", flush=True)
                        flAggregator.lock_user_waiting_queue.release()
                        time.sleep(2)
                        #time.sleep(10)

                    # 3. [FL] Aggregates the user models
                    collected_user_ids = [collected_user['id'] for collected_user in flAggregator.collected_users]

                    flAggregator.aggregate_models(r)
                    print(f"[Aggregator] [{R} ROUND] [{r}/{mec_round} round] ====== MEC model Aggregation using (Selected users: {len(flAggregator.selected_users)}, Collection fraction: {flAggregator.fraction}%)", flush=True)
                    print(f"[Aggregator] [{R} ROUND] [{r}/{mec_round} round] ====== Collected users ({len(collected_user_ids)} users)")  #{collected_user_ids})")

                    # 유저 별 학습에 사용한 데이터 수 (temp: dummy 값)
                    num_train_data_user = data_num_train / len(flAggregator.registered_users)
                    num_train_data_per_user = [num_train_data_user for _ in range(len(flAggregator.registered_users))]
                    # 유저 별 local model 학습 결과 accuracy (temp: dummy 값)
                    local_accuracy_per_user = [random.randint(1, 100) for _ in range(len(flAggregator.registered_users))]
                    # print(f"[Aggregator] [{R} ROUND] [{r} round] ====== Number of users' training data (temp):")  # {num_train_data_per_user}")
                    # print(f"[Aggregator] [{R} ROUND] [{r} round] ====== Accuracy of user models (temp):")  # {local_accuracy_per_user}")

                    # 4. [FL] Evaluates the mec model
                    results = flAggregator.mec_model.evaluate(x_test, y_test, batch_size=flAggregator.eval_batch, verbose=0)
                    print(f"[Aggregator] [{R} ROUND] [{r}/{mec_round} round] ====== MEC model Evaluation Accuracy: {round(results[1]*100, 2)}%", flush=True)

                    # 5. [FL] Write the results to tensorboard
                    with tf.summary.create_file_writer(tensorboard_log_dir).as_default():
                        tf.summary.scalar('loss', results[0], step=r)
                        tf.summary.scalar('accuracy', results[1], step=r)

                    time.sleep(fl_init_time)

                    # Clear
                    for _user in flAggregator.collected_users:
                        if f"{r}" in _user:
                            del (_user[f"{r}"])
                    flAggregator.lock_user_waiting_queue.acquire()
                    flAggregator.user_waiting_queue = set()
                    flAggregator.lock_user_waiting_queue.release()
                    flAggregator.collected_users = []
                    flAggregator.selected_users = []

                    # Calculate the elapsed time
                    mec_round_end_time = datetime.now()
                    mec_round_elapsed_time = mec_round_end_time - mec_round_start_time
                    print(f"[Aggregator] [{R} ROUND] [{r}/{mec_round} round] ====== Finish MEC round (round elapsed time: {round(mec_round_elapsed_time.total_seconds(), 2)} seconds)", flush=True)

                    # FL Info update
                    flAggregator.num_aggregated_users = len(collected_user_ids)
                    flAggregator.current_mec_round = r
                    flAggregator.current_accuracy = round(results[1]*100, 2)

                # 5. [FL] Evaluates the final mec model in each global round
                results = flAggregator.mec_model.evaluate(x_test, y_test, batch_size=flAggregator.eval_batch, verbose=1)
                mec_fl_end_time = datetime.now()
                mec_fl_elapsed_time = mec_fl_end_time - mec_fl_start_time

                print(f"[Aggregation] [{R} ROUND] ====== Finish MEC Federated Learning (MEC FL elapsed time: {round(mec_fl_elapsed_time.total_seconds(), 2)} seconds)")
                print(f"[Aggregation] [{R} ROUND] ====== Number of users: {num_users_mec}")
                print(f"[Aggregation] [{R} ROUND] ====== Final Accuracy: {results[1]}, Loss: {-1}")
                print(f"[Aggregation] [{R} ROUND] ====== Maximum Accuracy: {-1} round ({-1} round)")
                print(f"[Aggregation] [{R} ROUND] ====== First Reach round to target accuracy (97%): {-1} round (accuracy: {-1})")
                print(f"[Aggregation] [{R} ROUND] ====== Number of model sending to users: {mec_round * num_users_mec} (round x users)")
                print(f"[Aggregation] [{R} ROUND] ====== Total size of sent models to users: {mec_round * num_users_mec * 6.6}MB (round x users x model_size)")
                print(f"[Aggregation] [{R} ROUND] ====== Number of model receiving to users: {mec_round * num_users_mec} (round x users)")
                print(f"[Aggregation] [{R} ROUND] ====== Total size of received models to users: {mec_round * num_users_mec * 6.6}MB (round x users x model_size)")
                number_participation_per_user = [mec_round for _ in range(len(flAggregator.registered_users))]
                print(f"[Aggregation] [{R} ROUND] ====== Number of users' participation: ") # {number_participation_per_user}")
                total_trained_label_distribution = ["1M" for _ in range(len(flAggregator.registered_users))]
                print(f"[Aggregation] [{R} ROUND] ====== Total trained data's label distribution: ", flush=True) # {total_trained_label_distribution}")

                flAggregator.running_MEC_FL = False

                # 6. [FL] Uploads the MEC model to the Leader
                if flAggregator.is_hfl:
                    mec_model_weights = flAggregator.mec_model.get_weights()

                    # 7. [FL] Store Model weight to ipfs
                    if flAggregator.ipfs == True:
                        start_idx=r
                        agg_model_weight_file_name = ("agg_model_weights"+str(start_idx)+".npz")
                        agg_model_weight_file=np.savez(agg_model_weight_file_name,*mec_model_weights)

                        # Upload private model to IPFS
                        agg_model_weight_upload_start = time.time()
                        agg_model_weight_hash, private_model_weight_size = ipfs.upload_to_ipfs(agg_model_weight_file_name)
                        agg_model_weight_upload_end = time.time()
                        agg_model_weight_upload_time = agg_model_weight_upload_end - agg_model_weight_upload_start
                        print(f'>> Uploaded agg model weight to IPFS successfully. {agg_model_weight_upload_time}s', flush=True)
                        print(f'>> 해시값 확인 출력용. {agg_model_weight_hash}s', flush=True)
                        flAggregator.return_mec_model_hash(flAggregator.current_global_round,model_hash=agg_model_weight_hash)
                    else:
                        model_layers, model_layers_shape = helper.serialize_model_weights(mec_model_weights)
                        flAggregator.return_mec_model(flAggregator.current_global_round, model_layers, model_layers_shape)
                        
                else:
                    break
        for ip_port, userset in usersets.items():
            for user in userset:
                flAggregator.request_user_model_set(ip_port, userset, r, True)
        flAggregator.__init__()
        flAggregator.return_agg_condition(my_id)
        print("DONE")
