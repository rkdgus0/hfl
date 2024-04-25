import argparse
import configparser
import numpy as np
import grpc

import Library.helper as helper
import fl_pb2
import fl_pb2_grpc

MAX_MESSAGE_LENGTH = helper.MAX_MESSAGE_LENGTH

def ini_file_reader(file_name):
    ini_dict = {}
    config = configparser.ConfigParser()
    config.read(f'config/{file_name}.ini')
    for section in config.sections():
        ini_dict[section] = {}
        for option in config.options(section):
            ini_dict[section][option] = config.get(section, option)
    return ini_dict

def send_config_to_manager(config_dict, manager_ip, manager_port):
    print(f"#HFL#[Client] Client send Config file to Manager ({manager_ip}:{manager_port})", flush=True)
    with grpc.insecure_channel(f'{manager_ip}:{manager_port}',
                               options=[
                                    ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                               ]) as channel:
        stub = fl_pb2_grpc.ConfigServiceStub(channel)
        config_proto = fl_pb2.Config()
        for section, options in config_dict.items():
            section_proto = config_proto.sections[section]
            for key, value in options.items():
                section_proto.options[key] = value
        response=stub.SendConfig(config_proto)
        print(f"[Manager] Result of Client's config {response.ret_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='parameter reader')
    parser.add_argument('-m', '--manager_ip', dest='manager_ip', action='store',
                        type=str, required=True, help='<manager ip>:<port>')
    parser.add_argument('-f', '--file_name', dest='file_name', action='store',    
                        type=str, default='fl_config', help="config 파일 이름")
    args = parser.parse_args()

    # Manager 정보
    manager_ip = args.manager_ip
    manager_port = helper.MANAGER_PORT        
    if manager_ip.find(':') > 0:
        manager_port = manager_ip.split(':')[1]        
        manager_ip = manager_ip.split(':')[0]
        
    config_file = args.file_name
    full_dict = ini_file_reader(config_file)
    print(f"#HFL#[Client] (config={config_file})", flush=True)
    print(full_dict)
    send_config_to_manager(full_dict, manager_ip, manager_port)