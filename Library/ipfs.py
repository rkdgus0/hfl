import ipfshttpclient
import os
import random
import copy
import time
import tensorflow as tf
from scipy.sparse import csr_matrix
import numpy as np

class FL_IPFS():
    def __init__(self): 
        # Connect attempt ipfs and failure error output
        try:
            self.ipfs = self.connect_to_ipfs('127.0.0.1')
            print("Connected to IPFS successfully!")
        except Exception as e:
            print("Failed to connect to IPFS:", str(e))
            self.ipfs = None

        #DEV_base -> 아래는 기존 안양대 코드 참조
        #def __init__(self, fs_list):
        #self.ipfs = self.connect_to_ipfs(random.choice(fs_list))

    # Connect to IPFS
    def connect_to_ipfs(self, ip):
        url = '/ip4/' + ip + '/tcp/5001/http'
        ipfs = ipfshttpclient.connect(url, auth=("root", "root"))
        return ipfs

    # Save and load model to/from IPFS
    def upload_to_ipfs(self, file_name):
        file_size = os.path.getsize(file_name)
        file_hash = self.ipfs.add(file_name)['Hash']
        print(f'>>({file_name}) file size: {file_size} Bytes')
        os.remove(file_name)
        #os.remove -> 지금은 생성하고 바로 삭제
        return file_hash, file_size

    # Download file using hash(고유값)
    def download_from_ipfs(self, file_hash):
        print(f'>> Download file ({file_hash})')
        self.ipfs.get(file_hash)
        return file_hash
    
    # Load model weights from files in IPFS
    def load_model_weights_from_ipfs(model_weight_IPFS):
        loaded_weight_from_np = np.load(model_weight_IPFS, allow_pickle=True)  
        model_weights = [loaded_weight_from_np[layer_name] for layer_name in loaded_weight_from_np.files]
        return model_weights
    
    def save_gradient(self, gradient, top_n_percent, user_id):
        current_time = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"gradient_{user_id}_{current_time}.npz"
        
        sparse_gradient = []
        for grad in gradient:
            threshold = np.percentile(np.abs(grad), 100 - top_n_percent)
            grad_sparse = csr_matrix(grad * (np.abs(grad) > threshold))
            sparse_gradient.append(grad_sparse)
        
        np.savez(file_name, *sparse_gradient)
        return file_name
    
    def train_private_model(self, global_model, dataset, fl_static_param, fl_dynamic_param):
        # Extract hyperparameters from fl_init_param
        batch_size = fl_dynamic_param['batch_size']
        epochs = fl_dynamic_param['local_epoch']
        learning_rate = fl_dynamic_param['learning_rate']
        optimizer = tf.keras.optimizers.get({'class_name': fl_static_param['optimizer'], 'config': {'learning_rate': learning_rate}})
        loss_function = fl_static_param['loss_function']

        # Create a copy of the global model for local training
        private_model = copy.deepcopy(global_model)
        global_weights = global_model.get_weights()
        private_model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

        # Train the local model using local data
        history = private_model.fit(dataset['x_train'], dataset['y_train'], validation_data=(dataset['x_test'], dataset['y_test']), batch_size=batch_size, epochs=epochs, verbose=1)

        private_weights = private_model.get_weights()
        weight_differences = [np.subtract(global_weight, private_weight) for global_weight, private_weight in zip(global_weights, private_weights)]

        return private_model, history, weight_differences
