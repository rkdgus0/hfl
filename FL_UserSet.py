import argparse
import copy
import logging
import random
import time
import threading
import socket
from concurrent import futures

import google
import grpc
import tensorflow as tf
from tensorflow.python.keras import models, optimizers, losses, metrics

import fl_pb2
import fl_pb2_grpc

import Library.helper as helper
from Library.datasets import *
from Library.models import *
from Library.ipfs import FL_IPFS
from Library.node import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

import FL_Manager

# FL Userset
# -> Aggregator과 함께 MEC FL을 수행
# -> 매 라운드마다 Aggregator에게 요청을 받으면 학습을 수행하고 그 결과인 User model을 반환
# -> Userset은 N개의 User를 담당하는 Class이며, 1개의 Userset는 1개의 프로세스로 실행됨
# -> Userset은 n (<=N) 개의 User에 대한 학습 요청을 받으면, n개의 학습을 모두 수행한 후 n개의 user model을 한 번에 반환

# MAX_MESSAGE_LENGTH = 16777216  # 16 MB
# MAX_MESSAGE_LENGTH = 33554432  # 32 MB
# MAX_MESSAGE_LENGTH = 335544320 # 320 MB
# MAX_MESSAGE_LENGTH = 1006632960  # 960 MB
MAX_MESSAGE_LENGTH = helper.MAX_MESSAGE_LENGTH

# ProtoBuf 기반 gRPC 함수 정의
# -> protos/fl.proto 에서 정의한 RPC 함수를 사용하기 위해서 "AggregatorRpcService" 같이 Class 및 RPC 함수를 정의해야 함
# -> 1. protos/fl.proto 파일 작성
# -> 2. python -m grpc_tools.protoc -I./protos --python_out=. --pyi_out=. --grpc_python_out=. ./protos/fl.proto 로 컴파일
# -> 3. fl_pb2.py, fl_pb2.pyi, fl_pb2_grpc.py 파일이 생성됨
# -> 4. "AggregatorRpcService" Class 및 RPC 함수 정의


# class UserSetRpcService
# Userset RPC 함수 정의
# -> Userset은 RPC Server를 실행하여 외부에 RPC 함수를 제공
# -> Aggregator는 해당 RPC 함수를 통해 Userset과 통신하여 연합학습을 수행
class UserSetRpcService(fl_pb2_grpc.UserSet):
    def __init__(self, user_set):
        super().__init__()
        self.user_set: FlUserSet = user_set
        
    # RequestUserModelSet()
    # -> MEC FL round 시작용 함수
    # -> Aggregator는 Global FL의 매 라운드마다 해당 함수를 호출하여 Userset에게 user model을 요청함
    def RequestUserModelSet(self, req_msg, context):
        fl_round = req_msg.model.fl_round
        finish_condition = req_msg.model.finish
        self.user_set.current_mec_round = fl_round
        self.user_set.finish_condition = finish_condition
        model_msg = req_msg.model

        req_user_ids = []
        for addr in req_msg.address_set:
            req_user_ids.append(addr.id)
 
        print(f"[User] [{fl_round} round] Received Local Training Request from the aggregator: {self.user_set.connected_aggregator['id']} ({self.user_set.connected_aggregator['ip']}:{self.user_set.connected_aggregator['port']})")
        
        return_msg_set = []
        if self.user_set.running_local_training == False and finish_condition == False:
            self.user_set.apply_to_global_model(model_msg)
            self.user_set.running_local_training = True
            self.user_set.requested_user_ids = req_user_ids
            for _, user in self.user_set.users.items():
                user_addr = user.get_pb2_address()
                user.current_round = model_msg.fl_round
                return_msg_set.append(
                    fl_pb2.ReturnMsg(accepted=True, sender_addr=user_addr, fl_round=model_msg.fl_round, num_trained_data=model_msg.num_trained_data))
            return fl_pb2.ReturnSetMsg(return_set=return_msg_set)
        return fl_pb2.ReturnSetMsg(return_set=return_msg_set)

    def ConfigSending(self, request, context):
        config_dict = helper.convert_proto_to_dict(request)

        My_network = config_dict['network']
        self.user_set.my_index = My_network['index']

        Network_data = config_dict['send_network']
        self.user_set.aggregator_ip = Network_data['ip']
        self.user_set.aggregator_port = Network_data['port']
        
        FL_Service = config_dict['FL_Service']
        # ====== Participate_Info ======
        self.user_set.total_user = helper.int_default(FL_Service, 'user', 10)
        self.user_set.ipfs = helper.bool_default(FL_Service, 'ipfs')
        self.user_set.top_weight_save_percent = helper.int_default(FL_Service,'top_weight_save_percent', 100)
        self.user_set.learning_layer_mode = helper.bool_default(FL_Service, 'learning_layer_mode')

        # ====== Model/Data_Info ======
        self.user_set.model = helper.str_default(FL_Service, 'model', 'mcmahan2NN')
        self.user_set.pretrained = helper.bool_default(FL_Service, 'pretrained')
        self.user_set.optimizer = helper.str_default(FL_Service, 'optimizer', 'adam')
        self.user_set.base_model = define_model(self.user_set.model , self.user_set.pretrained)
        self.user_set.dataset = helper.str_default(FL_Service, 'dataset', 'mnist')
        self.user_set.split = helper.str_default(FL_Service, 'data_split', 'iid')
        self.user_set.alpha = helper.float_default(FL_Service, 'diri_alpha', 3)
        self.user_set.byzantine = helper.bool_default(FL_Service, 'byzantine')
        self.user_set.num_data_force = helper.int_default(FL_Service, 'force_num', 0)
        (x_train, y_train), (self.user_set.x_test, self.user_set.y_test) = load_dataset(self.user_set.dataset)
        self.user_set.x_train, self.user_set.y_train = split_data_set(self.user_set.split, x_train, y_train, self.user_set.total_user, self.user_set.alpha)

        # ====== Training_Info ======
        self.user_set.local_epoch = helper.int_default(FL_Service, 'local_epoch', 5)
        self.user_set.local_batch_size = helper.int_default(FL_Service, 'local_batch_size', 256)
        self.user_set.learning_rate = helper.float_default(FL_Service, 'local_learning_rate', 0.001)
        self.user_set.pre_user_per_uset = int(FL_Service['pre_user_per_uset'])
        self.user_set.user_per_uset = int(FL_Service['user_per_uset'])

        return google.protobuf.empty_pb2.Empty()


class FlUser:
    '''
        사용자 정보 관리 클래스    
    '''
    def __init__(self):
        self.account = ''   #사용자 ID or 계정
        self.address = {}   #사용자 주소
        self.current_epoch = 0
        self.current_accuracy = 0
        self.num_participation = 0  #사용자 학습 파라미터

    def get_pb2_address(self):
        return fl_pb2.AddressMsg(id=self.address['id'], ip=self.address['ip'], port=self.address['port'])

    def get_str_address(self):
        return f"{self.address['id']} ({self.address['ip']}:{self.address['port']})"

    def set_address(self, _id, _ip, _port):
        self.address['id'] = f"{_id}"
        self.address['ip'] = f"{_ip}"
        self.address['port'] = f"{_port}"

    def set_my_dataset(self, user_id, x_train, y_train):
        self.user_id = user_id           #사용자 ID
        self.x_train = x_train[user_id] # Shape : (유저당 데이터 수, 32, 32, 3)
        self.y_train = y_train[user_id] # Shape : (유저당 데이터 수, 10)
        self.num_trained_data = len(x_train[user_id])


class FlUserSet:
    def __init__(self):
        self.my_data_index = None
        self.address = {}
        self.global_model = None
        self.user_model = None
        self.users = {}
        self.grpc_server = None
        self.connected_aggregator = None
        self.requested_user_ids = []
        self.running_local_training = False
        self.finish_condition = False
        self.current_mec_round = 0
        self.num_userset_data = 0
        self.aggregator_ip = None
        self.aggregator_port = None
        self.model = None
        self.optimizer = None
        self.pretrained = None
        self.dataset = None
        self.base_model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.split = None
        self.alpha = None
        self.byzantine = None
        self.num_data_force = None
        self.local_epoch = None
        self.local_batch_size = None
        self.learning_rate = None
        self.total_user = None
        self.pre_user_per_uset = None
        self.user_per_uset = None
        self.ipfs = None    #IPFS 사용 여부
        self.learning_layer_mode = None
        self.top_weight_save_percent = None


    def check_config_file(self):
        while self.x_train is None:
            print("[Userset] Waiting for input parameter to be present by Client...")
            time.sleep(5)

    # rpc server 실행 함수
    def rpc_serve(self):
        my_port = self.address['port']
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                             options=[
                                 ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                 ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                 ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                             ])
        rpcService = UserSetRpcService(self)
        fl_pb2_grpc.add_UserSetServicer_to_server(rpcService, server)
        server.add_insecure_port(f'[::]:{my_port}')

        server.start()
        print(f"[UserSet] Rpc service started!, listening on {my_port}")

        self.grpc_server = server
        # server.wait_for_termination()

    def return_userset_condition(self, my_id):
        print(f"[Aggregator] Send aggregator's condition to the manager ({manager_ip}:{manager_port})")
        with grpc.insecure_channel(f'{manager_ip}:{manager_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.ManagerStub(channel)
            stub.ReturnCondition(fl_pb2.ConditionMsg(id=my_id, condition=True))

    # ===== Manager's RPC 함수 호출 / Manager 에게 자신을 등록하는 함수
    def register_userset_to_manager(self):
        print(f"[Userset] Send to the manager Userset ip,port ({manager_ip}:{manager_port})")
        with grpc.insecure_channel(f'{manager_ip}:{manager_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.ManagerStub(channel)
            my_addr = self.get_pb2_address()
            ret_msg = stub.RegisterUserSet(my_addr)
            self.ret_msg=ret_msg
            print(f"[Userset] Registration result: {ret_msg.accepted}")

    def register_aggregator(self,users,agg_ip,agg_port):
        print(f"[Userset] Send aggregator registration request to the manager ({manager_ip}:{manager_port})")
        self.fl_manager=FL_Manager.FlManager()
        self.fl_manager.register_to_aggregator(users,agg_ip,agg_port)
        self.register_aggregator_info(self.ret_msg.sender_addr)
            
    # ===== Aggregator's RPC 함수 호출 =====
    # Aggregator에게 학습 결과 모델인 user model을 return하는 함수
    def return_user_model_set(self, model_set):
        with grpc.insecure_channel(f'{self.aggregator_ip}:{self.aggregator_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.AggregatorStub(channel)
            ret_msg = stub.ReturnUserModelSet(fl_pb2.ModelSetMsg(model_set=model_set))
            print(f"[UserSet] [{ret_msg.fl_round} round] Upload {len(model_set)} users' models to the aggregator."
                  f"Result: {ret_msg.accepted}")
            #print(f"[Userset 모델셋 입력확인] 데이터 수: {ret_msg.num_trained_data}")

    # MEC FL를 위한 학습 전 이번 Round에서 Aggregator에게 받은 MEC Model을 적용하는 함수
    def apply_to_global_model(self, _model_msg):
        #Unist 가중치 로드 함수
        #model_weights = _model_msg.model_weights
        
        #ETRI 가중치 로드 함수
        layers = _model_msg.model_layers
        layers_shape = _model_msg.model_layers_shape
        model_weights = helper.deserialize_model_weights(layers, layers_shape)
        self.global_model.set_weights(model_weights)

    def register_aggregator_info(self, message):
        aggregator = {'id': message.id, 'ip': message.ip, 'port': message.port}
        self.connected_aggregator = aggregator

    def init_user_model(self):
        self.base_model.compile(optimizer=self.optimizer,
                                loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                                metrics=[metrics.SparseCategoricalAccuracy()])
        self.base_model.compute_output_shape(input_shape=(None, 32, 32, 3))
        self.global_model = copy.deepcopy(self.base_model)  # for saving the global model from an aggregator
        self.user_model = copy.deepcopy(self.base_model)  # for training

    def set_address(self, _id, _ip, _port):
        self.address['id'] = f"{_id}"
        self.address['ip'] = f"{_ip}"
        self.address['port'] = f"{_port}"

    def get_pb2_address(self):
        # print(f"id={self.address['id']}, ip={self.address['ip']}, port={self.address['port']}")
        return fl_pb2.AddressMsg(id=self.address['id'], ip=self.address['ip'], port=self.address['port'])

    def get_str_address(self):
        return f"{self.address['id']} ({self.address['ip']}:{self.address['port']})"

    # 해당 userset에 속하는 user들을 초기화
    def init_users(self):
        '''Userset에 속하는 user 정보 초기화 함수'''
        my_index = int(self.my_index) #1
        #TODO my_index 정보가 어떻게 설정되는가? 확인 필요 
        user_per_uset = int(self.pre_user_per_uset) #4
        for order_idx in range(self.user_per_uset):
            user_index = (user_per_uset * my_index) + order_idx
            user_id = f"u{user_index}"
            print(f'[FL_UserSet]User_index : {user_index}')
            self.users[user_id] = FlUser()
            self.users[user_id].set_address(user_id, self.address['ip'], self.address['port'])
            self.users[user_id].set_my_dataset(user_index, self.x_train, self.y_train)

    def get_requested_users(self, requested_user_ids):
        requested_users = []
        for user_id in requested_user_ids:
            if user_id in self.users:
                requested_users.append(self.users[user_id])
        return requested_users
    
    # 학습 수행
    def run_local_training(self, user):
        x_train = user.x_train
        y_train = np.argmax(user.y_train, axis=1)
        x_test = self.x_test
        y_test = np.argmax(self.y_test, axis=1)
        num_train = len(y_train)
        
        # Byzantine인 경우, 실제 갖고 있는 data의 1%로만 학습 수행 후 결과 반환
        if self.byzantine:
            num_train = num_train // 100
            x_train = (user.x_train)[:num_train] # Shape : (데이터 개수, 32, 32, 3)
            y_train = (user.y_train)[:num_train] # Shape : (데이터 개수, 10) -> (30000,)shape으로 받음
        
        if self.num_data_force != 0 and len(user.y_train) >= self.num_data_force:
            x_train = (user.x_train)[:self.num_data_force]
            y_train = (user.y_train)[:self.num_data_force]
        print(f"user: {user.address['id']} trains using {num_train} data. Byzantine: {self.byzantine}", flush=True)

        user.current_epoch = self.local_epoch
        user.current_batch_size = self.local_batch_size

        self.user_model.fit(x_train, y_train, batch_size=self.local_batch_size, epochs=self.local_epoch, verbose=0)

        if helper.LOCAL_EVALUATE:
            results = self.user_model.evaluate(x_test, y_test, batch_size=64)
            user.current_accuracy = round(results[1]*100, 2)
        user.num_trained_data = num_train
        print(f'[Userset] 데이터 개수 : {num_train}')
        return self.user_model, num_train


if __name__ == '__main__':
    #Library/Util.py에서 arg parser 가져오기
    parser = argparse.ArgumentParser(prog='Federated Learning Userset')
    parser.add_argument('-i', '--my_id', dest='my_id', action='store', type=str)
    parser.add_argument('-p', '--my_port', dest='my_port', action='store', type=str, default='50100')
    parser.add_argument('-m', '--manager_ip', dest='manager_ip', action='store',
                        type=str, required=True, help='<manager ip>:<port>')
    args = parser.parse_args()

    #학습 모델 저장을 위한 IPFS 호출
    ipfs = FL_IPFS()

    #Userset 통신정보
    my_ip = socket.gethostbyname(socket.gethostname())
    my_port = args.my_port
    if args.my_id is None:
        my_id = 'us_' + my_ip + my_port
    else:   
        my_id = args.my_id
    
    logging.basicConfig()
    flUserSet = None
    # GPU Dynamic allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    # Manager 정보
    manager_ip = args.manager_ip
    manager_port = helper.MANAGER_PORT
    if manager_ip.find(':') > 0:
        manager_port = manager_ip.split(':')[1]
        manager_ip = manager_ip.split(':')[0]
        
    print(f"#HFL#[GlobalAggregator] (id={my_id}, ip={my_ip}:{my_port}, manager={manager_ip}:{manager_port})", flush=True)
    
    # Initialize User
    while True:
        if flUserSet is None:
            flUserSet = FlUserSet()
        flUserSet.set_address(my_id, my_ip, my_port)
        flUserSet.rpc_serve()

        flUserSet.register_userset_to_manager()
        flUserSet.check_config_file()
        print('\n[Userset] We get the Config Datas!\n')

        flUserSet.init_user_model()  # every user uses this model for training and evaluation
        flUserSet.init_users()
        flUserSet.register_aggregator(flUserSet.users,flUserSet.aggregator_ip,flUserSet.aggregator_port)
        # USER들의 연합학습 코드
        while True:
            if flUserSet.finish_condition == True:
                flUserSet.__init__()
                flUserSet.return_userset_condition(my_id)
                break
            # print("[FL Client] Waiting for local training request...")
            time.sleep(1)
            if flUserSet.running_local_training:
                fl_round = flUserSet.current_mec_round

                requested_users = flUserSet.get_requested_users(flUserSet.requested_user_ids)
                remaining_users = requested_users
                NUM_MAX_USER_IN_MESSAGE = 100

                idx = 0
                start_idx = 0
                end_idx = NUM_MAX_USER_IN_MESSAGE
                if len(remaining_users) < NUM_MAX_USER_IN_MESSAGE:
                    end_idx = len(remaining_users)

                while start_idx < end_idx:
                    model_set = []
                    num_layer_split = 3
                    num_users = flUserSet.pre_user_per_uset
                    split_user = num_users // num_layer_split
                    total_cnt_top_weight = None
                    for user in remaining_users[start_idx:end_idx]:
                        user.num_participation += 1
                        print(f"[UserSet] [{fl_round} round] [User {idx+1}/{len(requested_users)}] Start Local Training", flush=True)
                        print(f"[UserSet] [{fl_round} round] [User {idx+1}/{len(requested_users)}] User ID: {user.address['id']}, Number of participation: {user.num_participation}")
                        print(f"[UserSet] [{fl_round} round] [User {idx+1}/{len(requested_users)}] Data: MNIST ({user.num_trained_data} samples), Data distribution: {flUserSet.split}")
                        print(f"[UserSet] [{fl_round} round] [User {idx+1}/{len(requested_users)}] Epochs: {flUserSet.local_epoch}, Training Batch size: {flUserSet.local_batch_size}")
                        print(f"[UserSet] [{fl_round} round] [User {idx+1}/{len(requested_users)}] Learning rate: {flUserSet.learning_rate}")
                        user_addr = user.get_pb2_address()

                        flUserSet.user_model.set_weights(flUserSet.global_model.get_weights())
                        user_model, num_trained_data= flUserSet.run_local_training(user)

                        #확인 출력용
                        #user_model.summary()
                        #print(user_model.get_weights())

                        global_model_weight = flUserSet.global_model.get_weights() #global 가중치 저장
                        local_train_weight = user_model.get_weights()
                        weight_differences = [np.subtract(global_weight, local_weight) for global_weight, local_weight in zip(global_model_weight, local_train_weight)]
                        #print(f"{len(global_model_weight)}####{len(local_train_weight)}####{len(weight_differences)}###")
                        #FL_Node.compare_model(flUserSet.global_model,flUserSet.user_model)

                        # 가중치 상위 10% 만 추출 후 나머지 0으로 치환
                        if flUserSet.top_weight_save_percent < 100:
                            occurrences = FL_Node.count_top_weight_occurrences_across_models(weight_differences,flUserSet.top_weight_save_percent/100)
    
                            # 라운드마다 선택된 가중치 누적으로 합산
                            total_cnt_top_weight = FL_Node.accumulate_occurrences([occurrences], total_cnt_top_weight)
                            user_model=FL_Node.retain_top_weights(user_model,weight_differences,flUserSet.top_weight_save_percent/100)
                            #print(len(user_model.get_weights()))
                            #mcmahan2nn 기준 전체 가중치 197432개
                            #user_model.summary()
                        
                        #TODO
                        # 레이어 별 가중치를 잘라서 agg에게 보냄
                        # mcmahan2nn 기준 layer (0,1),(2,3),(4,5) 로 3layer로 생각하여 작성
                        #FIXME 함수화
                        if flUserSet.learning_layer_mode == True:
                            if idx+1 <= split_user:
                                trained_model = FL_Node.set_weights_zero_except_one_layer(user_model, 0)
                            elif split_user*+1 <= idx+1 <= split_user * 2:
                                trained_model = FL_Node.set_weights_zero_except_one_layer(user_model, 1)
                            else:
                                trained_model = FL_Node.set_weights_zero_except_one_layer(user_model, 2)

                        elif flUserSet.learning_layer_mode == False:
                            trained_model = user_model
                        
                        # fl_config's [FL_service] 에서 ipfs == true 이면
                        if flUserSet.ipfs == True:

                            # 가중치 전체를 ipfs로 보냄
                            # Save private model / 가중치를 .npz 파일로 저장
                            weight = trained_model.get_weights()
                            local_model_weight_file_name = ("model_weights_"+str(start_idx)+"_"+str(user.address['id'])+".npz")
                            local_model_weight_file=np.savez(local_model_weight_file_name,*weight)
                            
                            # Upload private model to IPFS
                            local_model_weight_upload_start = time.time()
                            local_model_weight_hash, local_model_weight_size = ipfs.upload_to_ipfs(local_model_weight_file_name)
                            local_model_weight_upload_end = time.time()
                            local_model_weight_upload_time = local_model_weight_upload_end - local_model_weight_upload_start
                            print(f'>> Uploaded private model sparse matrix to IPFS successfully. {local_model_weight_upload_time}s', flush=True)
                            print(f'>> 해시값 확인 출력용. {local_model_weight_hash}s', flush=True)

                            model_set.append(fl_pb2.ModelMsg(fl_round=fl_round, sender_addr=user_addr,
                                                            num_trained_data=num_trained_data,
                                                            model_hash=local_model_weight_hash))
                                                            
                        # flUserSet.ipfs = false
                        else:
                            model_layers, model_layers_shape = helper.serialize_model_weights(user_model.get_weights())
                            model_set.append(fl_pb2.ModelMsg(fl_round=fl_round, sender_addr=user_addr,
                                                            model_layers=model_layers, model_layers_shape=model_layers_shape,
                                                            num_trained_data=num_trained_data,))
                                                            # num_trained_data=num_trained_data, num_trained_data_per_label=num_train_per_label,
                                                            # num_epoch=local_epoch))
                        idx += 1

                    print(f"[UserSet] [{fl_round} Round] Try to send {len(model_set)} users' models to the aggregator {flUserSet.connected_aggregator['id']} ({flUserSet.connected_aggregator['ip']}:{flUserSet.connected_aggregator['port']})", flush=True)
                    flUserSet.return_user_model_set(model_set)
                    print(f"[UserSet] [{fl_round} Round] Success to send {len(model_set)} users' models to the aggregator {flUserSet.connected_aggregator['id']} ({flUserSet.connected_aggregator['ip']}:{flUserSet.connected_aggregator['port']})", flush=True)
                    
                    start_idx = end_idx
                    end_idx = end_idx + NUM_MAX_USER_IN_MESSAGE
                    if len(remaining_users) < end_idx:
                        end_idx = len(remaining_users)

                flUserSet.running_local_training = False

    flClient.my_grpc_server.wait_for_termination()
    # flCleint.request_global_model()
