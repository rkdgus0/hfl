import copy
from tensorflow.python.keras import losses, metrics
from tensorflow.keras.optimizers import SGD
import random
import numpy as np

from Library.helper import *

class SERVER():
    def __init__(self, model, mecs, NUM_MEC, mec_delay, test_data, delay_method, eval_batch, hybrid, delay_range=0, 
                 agg_method=['Fedavg'], delay_epoch=0, n_epochs=1, model_decay='Equal', num_mec_datas=None):
        self.acc = 0
        self.model = model
        self.NUM_MEC = NUM_MEC
        self.mec_delay = copy.deepcopy(mec_delay)
        self.init_mec_delay = copy.deepcopy(tuple(mec_delay))
        self.delay_method = delay_method
        self.delay_range = delay_range
        self.mecs = mecs
        self.test_data = test_data
        self.agg_method = agg_method
        self.delay_epoch = delay_epoch
        self.n_epochs = n_epochs
        self.model_decay = model_decay
        self.eval_batch = eval_batch
        self.hybrid = hybrid
        self.num_mec_data = copy.deepcopy(num_mec_datas)
        self.model.compile(optimizer='adam',
                           loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=[metrics.SparseCategoricalAccuracy()])
        self.base_model = copy.deepcopy(model)

    def train(self):
        uploaded_models = []
        participating_mec = []
        for mec_idx in range(self.NUM_MEC):
            if self.mec_delay[mec_idx] == 0 or self.mec_delay[mec_idx] == None:
                local_epoch = self.calc_local_epoch(mec_idx)
                self.mecs.train(mec_idx, local_epoch)

                participating_mec.append(mec_idx)
                uploaded_models.append(copy.deepcopy(self.mecs.MEC_models[mec_idx])) # .get_weights())
                self.mec_delay[mec_idx] = self.set_mec_delay(mec_idx)
            else:
                self.mec_delay[mec_idx] -= 1

        print('Participating MEC: ', *participating_mec)
        if len(participating_mec) == 0:
            return
        
        final_ratio = None
        increase_ratio = []
        x_test, y_test = self.test_data
        y_test = np.argmax(y_test, axis=1).astype(int)
        if self.hybrid:
            for method in self.agg_method:
                agg_ratio = calc_avg_ratio(self.test_data, self.model, method, uploaded_models, participating_mec, self.num_mec_data)
                sub_w = average_model(uploaded_models, agg_ratio)
                sub_model = copy.deepcopy(self.base_model)
                sub_model.set_weights(sub_w)
                sub_results = sub_model.evaluate(x_test, y_test, batch_size=self.eval_batch, verbose=0)
                increase = sub_results[1] - self.acc
                print(f'[Simulator] Aggregation Method : {method}, Previous Acc : {self.acc}, Test Acc : {sub_results[1]}, Increase : {increase}')
                increase_ratio.append(increase)
                if final_ratio is None:
                    final_ratio = [agg_ratio]
                else:
                    final_ratio.append(agg_ratio)
                
                del sub_model
                sub_w.clear()
                #print(f'[Simulator] Avg Ratio : {final_ratio}')
            
            '''if all(value < 0 for value in increase_ratio):
                normalized_ratio = [1 for _ in range(len(increase_ratio))]
            else:
                normalized_ratio = [max(0, value) for value in increase_ratio]
            normalized_ratio = [val / sum(normalized_ratio) for val in normalized_ratio]
            print(f'[Simulator] Agg_ratio의 가중치 : {np.array(normalized_ratio)* 100}')
            final_ratio = [[element * normalized_ratio[k] for element in row] for k, row in enumerate(final_ratio)]
            print(f'가중치 적용 : {self.agg_method[max_index]} 제외 0.1')
            final_ratio = [[element if element.index == max_index else element * 0.1 for element in row] for k, row in enumerate(final_ratio)]
            final_ratio = np.sum(final_ratio, axis=0)'''
            
            max_index = increase_ratio.index(max(increase_ratio))
            final_ratio = final_ratio[max_index]
            print(f'적용 메소드 : {self.agg_method[max_index]}\n')
            avg_model = average_model(uploaded_models, final_ratio)
        else:
            method = self.agg_method[0]
            agg_ratio = calc_avg_ratio(self.test_data, self.model, method, uploaded_models, participating_mec, self.num_mec_data)
            avg_model = average_model(uploaded_models, final_ratio)

        self.model.set_weights(avg_model)
        for mec_idx in participating_mec:
            self.mecs.MEC_models[mec_idx] = copy.deepcopy(avg_model) # .set_weights(avg_model)

        uploaded_models.clear()
        avg_model.clear()

        return

    def set_mec_delay(self, mec_idx):
        if self.delay_method == 'Fixed':
            return self.init_mec_delay[mec_idx]# + 1
        elif self.delay_method == 'Range':
            dr = self.delay_range
            return max(0, self.init_mec_delay[mec_idx] + random.randint(-dr, dr))# + 1
        
    def test(self, model=None):
        test_x, test_y = self.test_data
        test_y = np.argmax(test_y, axis=1).astype(int)
        if model == None:
            return self.model.evaluate(test_x, test_y, batch_size=self.eval_batch)
        else:
            return model.evaluate(test_x, test_y, batch_size=self.eval_batch)
        
    def calc_local_epoch(self, mec_idx):
        if self.delay_epoch == 0:
            return self.n_epochs # self.init_mec_delay[mec_idx]
        else:
            return max(1, self.init_mec_delay[mec_idx] * self.delay_epoch)
    def set_lr(self, lr):
        self.mecs.clients.set_lr(lr)


class MEC():
    def __init__(self, model, mec_user_mapping, clients):
        super(MEC, self).__init__()
        self.mec_user_mapping = mec_user_mapping
        self.MEC_models = [model.get_weights() for _ in range(len(mec_user_mapping))]
        self.clients = clients
        print(f'MEC_MODEL 개수 : {len(mec_user_mapping)}')

    def train(self, mec_id, local_epoch):
        MEC_MODEL = copy.deepcopy(self.MEC_models[mec_id])
        clients = self.mec_user_mapping[mec_id]
        avg_models = []

        for client_idx in clients:
            self.clients.train(client_idx, MEC_MODEL, local_epochs=local_epoch)
            avg_models.append(copy.deepcopy(self.clients.model.get_weights()))
        self.MEC_models[mec_id] = copy.deepcopy(average_model(avg_models))

        del MEC_MODEL
        avg_models.clear()

        return

# TODO Model compile 수정(learning_rate, decay method 동작 X) 필요
class USER():
    def __init__(self, lr, model, x_dataset, y_dataset, epochs, batch_size):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = copy.deepcopy(model)
        self.model.compile(optimizer='adam',
                           loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=[metrics.SparseCategoricalAccuracy()])
        '''self.model.compile(loss=losses.SparseCategoricalCrossentropy(),
                           optimizer=SGD(learning_rate=lr, clipvalue=1.0))'''

    def train(self, client_idx, model_parameters, local_epochs=1):
        self.model.set_weights(model_parameters)
        y_train = np.argmax(self.y_dataset[client_idx], axis=1).astype(int)
        self.model.fit(self.x_dataset[client_idx], y_train,
                       epochs=local_epochs, batch_size=self.batch_size, verbose=0)
        return

    def set_lr(self, lr):
        self.model.compile(optimizer='adam',
                           loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=[metrics.SparseCategoricalAccuracy()])

        '''self.model.compile(optimizer=SGD(learning_rate=lr),
                                        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                                        metrics=[metrics.SparseCategoricalAccuracy()])'''