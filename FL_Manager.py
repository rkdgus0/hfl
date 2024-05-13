import configparser
import argparse
import grpc
#import sys
import fl_pb2
import fl_pb2_grpc
import google

from concurrent import futures
from Library.helper import *

import socket
import time

MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
print(MAX_MESSAGE_LENGTH)

LOG_HEAD = '@FL [MANAGER]'

#서브 셋 가져오는 함수
def proto_to_dict(proto_message, target_sections=None):
    result_dict = {}

    for section_name, section_proto in proto_message.sections.items():
        if target_sections is None or section_name in target_sections:
            result_dict[section_name] = {}
            for option_name, option_value in section_proto.options.items():
                result_dict[section_name][option_name] = option_value

    return result_dict

class ManagerRpcService(fl_pb2_grpc.Manager):
    def __init__(self, manager):
        super().__init__()
        self.manager: FlManager = manager

    def RegisterLeader(self, addr_msg, context):  # context.peer
        print(
            f"{LOG_HEAD} Call RegisterLeader [ID: {addr_msg.id}, IP: {addr_msg.ip}, Port: {addr_msg.port}]")
        self.manager.register_leader(addr_msg)
        my_addr = self.manager.get_pb2_address()
        return fl_pb2.ReturnMsg(accepted=True, sender_addr=my_addr)
    
    def RegisterAggregator(self, addr_msg, context):  ####
        print(
            f"{LOG_HEAD} Call RegisterAggregator [ID: {addr_msg.id}, IP: {addr_msg.ip}, Port: {addr_msg.port}]")
        self.manager.register_aggregator(addr_msg)
        my_addr = self.manager.get_pb2_address()
        return fl_pb2.ReturnMsg(accepted=True, sender_addr=my_addr)
    
    def RegisterUserSet(self, addr_msg, context):  ####
        print(
            f"{LOG_HEAD} Call RegisterUserSet [ID: {addr_msg.id}, IP: {addr_msg.ip}, Port: {addr_msg.port}]")
        self.manager.register_userset(addr_msg)
        my_addr = self.manager.get_pb2_address()
        return fl_pb2.ReturnMsg(accepted=True, sender_addr=my_addr)
    
    def ReportFinalResults(self, final_results_msg, context):
        #TODO 리더만 호출하는 것은 아님, 확인 및 변경 필요
        print(
            f"{LOG_HEAD} Call ReportFinalResults from Leader. Losses: {final_results_msg.loss}, Accuracies: {final_results_msg.accuracy}, Elapsed Time: {final_results_msg.elapsed_time}", flush=True)
        self.manager.register_total_result(final_results_msg)
        return fl_pb2.ReturnMsg(accepted=True)
    
    def ReturnCondition(self, condition_msg, context):
        net_config = configparser.ConfigParser()
        net_config.read('config/fl_net_config.ini')
        id = condition_msg.id
        exp_name = condition_msg.exp_name
        finish_condition = condition_msg.condition
        if finish_condition:
            for section in net_config.sections():
                if config.get(section, 'id') == id:
                    found_section = section
                    break
            num_used = net_config[found_section].getint('current_used')
            value = max(0, num_used - 1)
            net_config[found_section].update({'current_used' : f'{value}'})
            print('\n'f'Exp : {exp_name} is finished! and {id} is reconnected!')

        with open('config/fl_net_config.ini', 'w', encoding='utf-8') as file:
            net_config.write(file)
        
        return fl_pb2.ReturnMsg(accepted=True)


class ConfigService(fl_pb2_grpc.ConfigServiceServicer):
    def __init__(self):
        self.manager_dict = {}
        self.leader_dict = {}
        self.aggregator_dict = {}
        self.Userset_dict = {}
        
    def get_using_module_index(self, net_config, module_name, total_num, collect_num):
        result_dict = {}
        order_num = 0
        for idx in range(total_num):
            if net_config.has_section(f"{module_name}{idx}"):
                net_section = net_config[f"{module_name}{idx}"]
                current_used = net_section.getint('current_used', fallback = 0)
                sub_id = net_section.get('id')
                sub_ip = net_section.get('ip')
                sub_port = net_section.get('port')
                if current_used < 1 and module_name !='userset':
                    net_config.set(f'{module_name}{idx}', 'current_used', f'{current_used + 1}')
                    result_dict[f'{order_num}'] = {'id': f'{sub_id}', 'ip': f'{sub_ip}', 'port': f'{sub_port}', 'index': f'{order_num}'}
                    order_num += 1
                elif current_used < 1 and module_name == 'userset':
                    net_config.set(f'{module_name}{idx}', 'current_used', f'{current_used + 1}')
                    result_dict[f'{order_num}'] = {'id': f'{sub_id}', 'ip': f'{sub_ip}', 'port': f'{sub_port}', 'index': f'{order_num}'}
                    order_num += 1
            if order_num == collect_num:
                with open(f'config/fl_net_config.ini', 'w', encoding='utf-8') as file:
                    net_config.write(file)
                return result_dict
        
    def check_online(self, simul, n_leader, n_agg, n_uset):
        leader_index = None
        agg_index = None
        uset_index = None
        net_config = configparser.ConfigParser()
        net_config.read('config/fl_net_config.ini')
        if n_leader > 0:
            leader_count = count_module(net_config, 'leader', n_leader, online=1)
            leader_index = self.get_using_module_index(net_config, 'leader', leader_count, n_leader)
        if not simul:
            aggregator_count = count_module(net_config, 'aggregator', n_agg, online=1)
            userset_count = count_module(net_config, 'userset', n_uset, online=1)
            agg_index = self.get_using_module_index(net_config, 'aggregator', aggregator_count, n_agg)
            uset_index = self.get_using_module_index(net_config, 'userset', userset_count, n_uset)
            
        return leader_index, agg_index, uset_index

    def send_dict_data_sub(self, stub, config_dict):
        sending_data = fl_pb2.Config()
        for section1, options1 in config_dict.items():
            section_proto = sending_data.sections[section1]
            for key1, value1 in options1.items():
                section_proto.options[key1] = value1
        if sending_data is not None:
            stub.ConfigSending(sending_data)

    def send_dict_data(self, path, config_dict, my_net_dict, upper_net_dict=None):
        idx = 0
        for order, section in my_net_dict.items():
            config_dict['network'] = my_net_dict[f'{int(order)}']
            with grpc.insecure_channel(f"{section['ip']}:{section['port']}",
                                       options=[('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]) as channel:
                if path == 'leader':
                    stub = fl_pb2_grpc.LeaderStub(channel)
                elif path == 'aggregator':
                    stub = fl_pb2_grpc.AggregatorStub(channel)
                    if upper_net_dict is not None:
                        config_dict['send_network'] = upper_net_dict['0']
                    config_dict['FL_Service'].update({'user_per_agg': f'{self.user_agg[idx]}'})
                    config_dict['network'].update({'index' : f'{idx}'})
                    print(f'[Manager] user_per_agg: {self.user_agg[idx]}')
                elif path == 'userset':
                    stub = fl_pb2_grpc.UserSetStub(channel)
                    order_idx = idx % len(upper_net_dict)
                    key_idx = list(upper_net_dict.keys())
                    config_dict['FL_Service'].update({'user_per_uset' : f'{self.user_uset[idx]}'})
                    config_dict['network'].update({'index' : f'{idx}'})
                    config_dict['send_network'] = upper_net_dict[f'{key_idx[order_idx]}']
                    print(f'[Manager] user_per_userset: {self.user_uset[idx]}')
                else:
                    raise ValueError("[Manager] check the path(leader, aggregator, userset) input")
                idx += 1
                self.send_dict_data_sub(stub, config_dict)
    
    def define_user_index(self, n_agg, n_userset, n_user):
        pre_user_agg = [[] for _ in range(n_agg)]
        user_per_uset = n_user // n_userset
        mod_user_uset = n_user % n_userset

        user_uset = [user_per_uset + 1 if i < mod_user_uset else user_per_uset for i in range(n_userset)]

        for j, set in enumerate(user_uset):
            mod = j % n_agg
            pre_user_agg[mod].append(set)
        user_agg = [sum(pre_user_agg[k]) for k in range(n_agg)]
        
        return user_uset, user_agg

    def SendConfig(self, request, context):
        print(f'[Manager] Request (Client:{context.peer()[5:]})')
        manager_dict = proto_to_dict(request, ['FL_Service'])
        print(f'[Manager] manager_dict:\n {manager_dict}')
        
        n_leader, n_agg, n_user, n_userset, simulate, ipfs, top_weight_save_percent, learning_layer_mode = self.extract_config_values(manager_dict)

        user_per_uset = n_user // n_userset
        is_hfl = n_leader > 0 and n_agg > 0

        try:
            leader_net, agg_net, uset_net = self.check_online(simulate, n_leader, n_agg, n_userset)
            config_dict = self.build_config_dict(request, n_userset, is_hfl, user_per_uset, ipfs, top_weight_save_percent, learning_layer_mode)
            if not simulate:
                self.user_uset, self.user_agg = self.define_user_index(n_agg, n_userset, n_user)
                if is_hfl:
                    self.send_dict_data('leader', config_dict, leader_net)
                self.send_dict_data('aggregator', config_dict, agg_net, leader_net)
                self.send_dict_data('userset', config_dict, uset_net, agg_net)
                
            else:
                self.send_dict_data('leader', config_dict, leader_net)
            
            result_msg = "Accept"
            print(f'\n[Manager] Federated Learning is intended to learn with (Leader : {n_leader}, Aggregator : {n_agg}, User : {n_user}), HFL : {is_hfl}) (Simulation? : {simulate})')
        
        except Exception as e:
            print(f'[Manager] You Should check Request')
            print(e)
            result_msg = str(e)
              
        google.protobuf.empty_pb2.Empty()
        return fl_pb2.ReturnConfigMsg(ret_msg=result_msg)
    
    def extract_config_values(self, manager_dict):
        n_dict = manager_dict['FL_Service'] 
        n_leader = int_default(n_dict, 'leader', 0)
        n_agg = int_default(n_dict, 'aggregator', 1)
        n_user = int_default(n_dict, 'user', 10)
        n_userset = min(int_default(n_dict, 'userset', 2), n_user)
        simulate = bool_default(n_dict, 'simulate')
        ipfs = bool_default(n_dict, 'ipfs')
        top_weight_save_percent = int_default(n_dict, 'top_weight_save_percent', 100)
        learning_layer_mode = bool_default(n_dict, 'learning_layer_mode')

        return n_leader, n_agg, n_user, n_userset, simulate, ipfs, top_weight_save_percent, learning_layer_mode
    
    def build_config_dict(self, request, n_userset, is_hfl, user_per_uset, ipfs, top_weight_save_percent, learning_layer_mode):
        config_dict = proto_to_dict(request)
        config_dict['FL_Service'].update({'n_userset': f'{n_userset}'})
        config_dict['FL_Service'].update({'is_hfl': f'{is_hfl}'})
        config_dict['FL_Service'].update({'pre_user_per_uset' : f'{user_per_uset}'})
        config_dict['FL_Service'].update({'ipfs': f'{ipfs}'})
        config_dict['FL_Service'].update({'top_weight_save_percent': f'{top_weight_save_percent}'})
        config_dict['FL_Service'].update({'learning_layer_mode': f'{learning_layer_mode}'})
        return config_dict
        


class FlManager:
    def __init__(self):
        self.info = ''
        self.address = {}
        self.grpc_server = None
        
        self.num_leader = 0                     #등록된 GlobalAggregator 수
        self.num_aggregator = 0                 #등록된 Aggregator 수
        self.num_userset = 0                    #등록된 UserSet 수
        self.registered_leader_info = []        #등록된 GlobalAggregator List
        self.registered_aggregators_info = []   #등록된 Aggregator List
        self.registered_userset_info = []       #등록된 UserSet List
        
        self.result_idx=0
        self.total_result=[]

    def set_address(self, _id, _ip, _port):
        self.address['id'] = _id
        self.address['ip'] = _ip
        self.address['port'] = _port

    def get_pb2_address(self):
        _id = self.address['id']
        _ip = self.address['ip']
        _port = self.address['port']

        return fl_pb2.AddressMsg(id=_id, ip=_ip, port=_port)
    
    def register_leader(self, message):
        new_leader = [message.id, message.ip, message.port]  # 2차원 리스트로 변경
        if new_leader not in self.registered_leader_info:
            self.registered_leader_info.append(new_leader)
            self.num_leader += 1
        self.print_network()

    def register_aggregator(self, message):
        new_aggregator = [ message.id, message.ip, message.port]
        if new_aggregator not in self.registered_aggregators_info:
            self.registered_aggregators_info.append(new_aggregator)
            self.num_aggregator += 1
        self.print_network()

    def register_userset(self, message):
        new_userset = [message.id, message.ip, message.port]  # 2차원 리스트로 변경
        if new_userset not in self.registered_userset_info:
            self.registered_userset_info.append(new_userset)
            self.num_userset += 1
        self.print_network()

    def register_total_result(self, message):
        result_entry = [
            self.result_idx,
            message.g_round,
            message.accuracy,
            message.loss,
            message.num_aggs,
            message.elapsed_time
        ]
        self.total_result.append(result_entry)
        self.result_idx += 1
        self.print_result()
        
    ### 여기에서 LeaderRpc RegisterAggregator를 호출해야함
    #input : network 정보 기반
    def register_to_leader(self,leader_ip,leader_port,agg_id,agg_ip,agg_port):
        print(f"#[Manager] Manager registers aggregator with Leader ({leader_ip}:{leader_port})")
        with grpc.insecure_channel(f'{leader_ip}:{leader_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.LeaderStub(channel)
            self.set_address(agg_id,agg_ip,agg_port)
            my_addr = self.get_pb2_address()
            try:
                ret_msg = stub.RegisterAggregator(my_addr)
                print(f"[Aggregator] Registration result: {ret_msg.accepted}")
                self.register_leader(ret_msg.sender_addr)
            except grpc.RpcError as e:
                print(f"Error during registration with Leader: {e}", flush=True)   
     
    def register_to_aggregator(self,users,agg_ip,agg_port):
        print(f"#[Manager] Manager registers Userset with Aggregator ({agg_ip}:{agg_port})", flush=True)
        with grpc.insecure_channel(f'{agg_ip}:{agg_port}',
                                   options=[
                                       ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                       ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                   ]) as channel:
            stub = fl_pb2_grpc.AggregatorStub(channel)

            user_address_set = []
            for _, user in users.items():
                user_address_set.append(user.get_pb2_address())
                
            try:
                ret_msg = stub.RegisterUserSet(fl_pb2.AddressSetMsg(address_set=user_address_set))
                print(f"[UserSet] Registration result: {ret_msg.accepted}")
            except grpc.RpcError as e:
                print(f"Error during registration: {e}", flush=True)


    def print_network(self):
        #print(f"\n\n{time.ctime()}")  
        t = time.localtime()
        current_time = time.strftime("%Y.%m.%d %H:%M:%S", t)
        print(f"\n{current_time}")     
        print(f"\n{LOG_HEAD} Waiting for register...\n\nManager: "
              f"{self.info}", flush=True)
        print(" -Global Aggreagtor(Leader) registered: "
              f"{self.num_leader}", flush=True)
        for leader in self.registered_leader_info:
            print(f"[{leader[0]}] {leader[1:]}")
        print(" -Aggregator registered: "
              f"{self.num_aggregator}", flush=True)
        for aggregator in self.registered_aggregators_info:
            print(f"[{aggregator[0]}] {aggregator[1:]}")
        print(" -Userset registered: "
              f"{self.num_userset}", flush=True)
        for userset in self.registered_userset_info:
            print(f"[{userset[0]}] {userset[1:]}")

    def print_result(self):
        print(f"\n\n{LOG_HEAD} Total Results:")
        for result_entry in self.total_result:
            result_idx, g_round, global_accs, global_losses, num_aggs, elapsed_time = result_entry
            print(f"{LOG_HEAD} === Result of Request #{result_idx + 1} ======", flush=True)
            print(f"{LOG_HEAD} [#{result_idx + 1}] ==== Leader Round: {g_round}", flush=True)
            print(f"{LOG_HEAD} [#{result_idx + 1}] ==== Finish Federated Learning ( FL elapsed time: {elapsed_time} seconds)", flush=True)
            print(f"{LOG_HEAD} [#{result_idx + 1}] ==== Final Accuracy: {global_accs}, Loss: {global_losses}", flush=True)
            print(f"{LOG_HEAD} [#{result_idx + 1}] ==== Number of model sending to aggregators: {g_round * num_aggs} (round x aggregators)", flush=True)
    
    def set_network_info(self, config, path, key, index):
        section = f'{path}{index}'
        if config.has_section(section):
            pass
        else:
            config.add_section(section)
        config[section].update({'id': key[0], 'ip': key[1], 'port': key[2]})
    
    def print_network_to_file(self, file_name):
        config = configparser.ConfigParser()
        config.read(f'config/{file_name}.ini')
        # leader 정보 추가
        for k, leader in enumerate(self.registered_leader_info):
            self.set_network_info(config, 'leader', leader, k)
        # aggregator 정보 추가
        for k, aggregator in enumerate(self.registered_aggregators_info):
            self.set_network_info(config, 'aggregator', aggregator, k)
        # userset 정보 추가
        for k, userset in enumerate(self.registered_userset_info):
            self.set_network_info(config, 'userset', userset, k)
        with open(f'config/{file_name}.ini', 'w', encoding='utf-8') as file:
            config.write(file)
    
    # ===== RPC server =====
    def rpc_serve(self):
        my_port =  self.address['port']
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                             options=[
                                 ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                 ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                 ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                             ])
        rpcService = ManagerRpcService(self)
        fl_pb2_grpc.add_ManagerServicer_to_server(rpcService, server)
        fl_pb2_grpc.add_ConfigServiceServicer_to_server(ConfigService(), server)
        server.add_insecure_port(f'[::]:{my_port}')

        server.start()
        print(f"{LOG_HEAD} Rpc service started!, listening on {my_port}")
        self.grpc_server = server
        # server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Federated Learning Manager')
    parser.add_argument('-m', '--manager_ip', dest='manager_ip', action='store',
                        type=str, help='<manager ip>:<port>')
    parser.add_argument('-f', '--file_name', dest='file_name', action='store', type=str,   
                        default='fl_net_config', help="fl network config를 저장하기 위한 파일 [경로]이름")
    parser.add_argument('-l', '--loop_time', dest='loop', action='store', type=int, default=5)
    args = parser.parse_args()
    
    if args.manager_ip is None:
        my_ip = socket.gethostbyname(socket.gethostname())
        my_port = MANAGER_PORT
    else:
        if args.manager_ip.find(':') > 0:
            my_port = args.manager_ip.split(':')[1]
            my_ip = args.manager_ip.split(':')[0]            
        else:
            my_ip = args.manager_ip
    
    my_id = 'Manager'
    loop = args.loop
    
    config = configparser.ConfigParser()
    file_name = args.file_name  #서비스 네트워크를 저장하는 파일
    with open(f'config/{file_name}.ini', 'w', encoding='utf-8') as file:
        config.write(file)
    #TODO 기존 파일 존재할때 Overwrite 또는 네트워크 복원 관련 기능 추가 필요
    
    print(f"{LOG_HEAD} (ip={my_ip}:{my_port}, refresh={loop}, net_file={file_name})", flush=True)
    # Initialize Manager
    flmanager = FlManager()
    flmanager.info = my_ip + ':' + str(my_port)
    flmanager.set_address(my_id, my_ip, my_port)
    flmanager.rpc_serve()
    print(f"{LOG_HEAD} Manager Service Start \n\t  waiting leaders, aggregators, usersets ...")

    while True:
        #TODO 변경이 발생한 경우만 기록되도록 변경 필요
        flmanager.print_network_to_file(file_name)
        #flmanager.print_network()
        t = time.localtime()
        current_time = time.strftime("%Y.%m.%d %H:%M:%S", t)
        print(f"\r{LOG_HEAD} Manager Waitting : {current_time}", end="")
        time.sleep(loop)
        