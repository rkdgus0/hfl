import sys
from collections import defaultdict

sys.path.append("../")  # To use fedlab library

import numpy as np
from keras.layers import *
import keras
from keras.datasets import cifar10, mnist
import tensorflow as tf
import random

LABELS_PER_USER = 2

random.seed(42)
np.random.seed(42)
# Define the TensorFlow data augmentation and normalization pipeline
def transform_train(image, label):
    # Random crop
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.random_crop(image, size=[50000, 32, 32, 3])

    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Convert to tensor


    # Normalize
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    image = (image - mean) / std

    return image, label


def transform_test(image, label):
    # Convert to tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Normalize
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    image = (image - mean) / std

    return image, label

def load_dataset(data):
    #TODO Dataset Cifar10에 대해 preprocessing 추가.
    if data == 'mnist':
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        x_train = np.expand_dims(x_train, axis=-1)
        x_train = np.repeat(x_train, 3, axis=-1)
        x_train = tf.image.resize(x_train, [32, 32])  # if we want to resize

        x_test = np.expand_dims(x_test, axis=-1)
        x_test = np.repeat(x_test, 3, axis=-1)
        x_test = tf.image.resize(x_test, [32, 32])  # if we want to resize

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    
    elif data == 'cifar10':
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def split_data_set(mode, x_train, y_train, n_user, alpha):
    if mode == 'random':
        return random_split_data(x_train, y_train, n_user)
    elif mode == 'iid':
        return iid_split_data(x_train, y_train, n_user)
    elif mode == 'non_iid':
        return non_iid_split_data(x_train, y_train, n_user)
    elif mode == 'diri':
        return diri_split_data(x_train, y_train, n_user, alpha)
    elif mode == 'diff':
        return diff_split_data(x_train, y_train, n_user)

#SIMULATION용 데이터 분산
def split_data(mode, origin_data, n_user, alpha):
    x , y = origin_data
    x_train, y_train = split_data_set(mode, x, y, n_user, alpha)
    client_datasets = []
    for i in range(n_user):
        client_datasets.append({'x':x_train[i],'y':y_train[i]})
    return client_datasets


def random_split_data(x_train, y_train, n_user): #유저 4명이라 가정.
    x, y = x_train, y_train # x : (60000, 32, 32, 3), y : (60000, 10)
    data_idx = [i for i in range(len(x))] # [1, 2, 3, ... , 59999]
    num_client_data = len(x) // n_user #몫만 가져옴, 60000/4 = 15000

    x_datasets = [[] for _ in range(n_user)] #유저 수만큼 생성 [[],[],[],[]]
    y_datasets = [[] for _ in range(n_user)] #유저 수만큼 생성 [[],[],[],[]]
    user_idx_list = [[] for _ in range(n_user)] # [[],[],[]...[]], shape : (유저 수, )인 2차원 배열 생성

    for i in range(n_user):
        user_samples = np.random.choice(data_idx, num_client_data, replace=False)
        data_idx = list(set(data_idx)- set(user_samples))
        user_idx_list[i].extend(user_samples)
        x_datasets[i] = np.take(x, user_idx_list[i], axis=0)
        y_datasets[i] = np.take(y, user_idx_list[i], axis=0)
        
    return x_datasets, y_datasets


def iid_split_data(x_train, y_train, n_user):
    x, y = x_train, y_train # x : (60000, 32, 32, 3), y : (60000, 10)

    x_datasets = [[] for _ in range(n_user)]
    y_datasets = [[] for _ in range(n_user)]

    tmp_y = np.argmax(y, axis=1) # 행축을 따라, 가장 높은 값 반환 shape : (60000,10) -> (60000,)
    #print(tmp_y.shape)

    unique_labels = np.unique(tmp_y, return_counts=False) # 라벨 중, 중복 값 삭제 [0,1,2,3,4,5,6,7,8,9]
    user_idx_list = [[] for _ in range(n_user)] # [[],[],[]...[]], shape : (유저 수, )인 2차원 배열 생성

    for label in unique_labels: #y_train의 인덱스만큼 반복 10회 반복
        label_indices = np.where(tmp_y == label)[0] #unique 라벨의 index를 반환 (라벨값이 같은 것들의 인덱스값들의 모음) ->[0]으로 인해 1차원 배열으로 반환
        data_per_user = len(label_indices) // n_user #해당 클래스를 몇개 가지고 있는지 
        for i in range(n_user): #유저 수만큼 반복
            user_samples = np.random.choice(label_indices, data_per_user, replace=False) #해당 클래스를 유저마다 고르게 분포하기 위해, 해당 클래스를 유저수에 맞게 choice함
            label_indices = list(set(label_indices) - set(user_samples)) #choice된 클래스는 label_indices에서 삭제
            user_idx_list[i].extend(user_samples) #user_idx_list의 유저번호에 해당 클래스의 index를 추가
    #반복문이 끝나면, 각 유저별로 동등하게 데이터의 인덱스가 2차원 배열로 저장됨 ex. [[4,13,45(0번 클래스),6030,9840(1번 클래스), ... ](0번 유저),[]]
    for i in range(n_user):
        x_datasets[i] = np.take(x, user_idx_list[i], axis=0)
        y_datasets[i] = np.take(y, user_idx_list[i], axis=0)
    return x_datasets, y_datasets


def non_iid_split_data(x_train, y_train, n_user): #예시는 MNIST, 유저 수 4명 기준
    np.random.seed(0)
    x, labels = x_train, y_train # x : (60000, 32, 32, 3), y : (60000, 10)
    
    arg_labels = np.argmax(labels, axis=1) # y : (60000,10) -> (60000,) 클래스 번호가 있는 1차원 배열
    arg_labels = arg_labels.reshape(1, -1) # arg_labels : (60000,) -> (1, 60000) 으로 행 벡터로 변환

    classes_per_client = LABELS_PER_USER
    x_datasets = [[] for _ in range(n_user)] # x_datasets : [[],[],[],[]]
    y_datasets = [[] for _ in range(n_user)] # y_datasets : [[],[],[],[]]

    num_shards = classes_per_client * n_user # num_shards : 2 * 4 = 8개
    num_imgs = int(len(x) / num_shards) # 60000 / 8(샤드 수) = 7500개
    idx_shard = [i for i in range(num_shards)] # idx_shard : [0, 1, 2, ..., 7] 샤드 인덱스
    dict_users = [[] for _ in range(n_user)] # [[],[],[]...[]], shape : (유저 수, )인 2차원 배열 생성
    idxs = np.arange(num_shards * num_imgs) # idxs : array([0, 1, 2, 3, ... , 59999])
    
    #     no need to judge train ans test here
    idxs_labels = np.vstack((idxs, arg_labels)) # [[0, 1], [1, 1], [2, 4], [3, 9], ... ,[59999, 3]] shape : (60000,2)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] 
    #idxs_labels = idxs_labels[idxs_labels[:, 1].argsort(), :] #[[0, 1], [1, 1], ... , [59999, 3], ... , [2, 4], ... , [3, 9]]
    idxs = idxs_labels[0, :]  # idxs :[[0], [1], ... , [59999], ... , [2], ... , [3]]
    #idxs = [item[0] for item in idxs]
    idxs = idxs.astype(int) #int값으로 변환

    for i in range(n_user): #유저 수만큼 반복 (4번 반복)
        rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # 샤드 개수 8개 중에 2개를 비 복원 추출 ex. 2, 4
        idx_shard = list(set(idx_shard) - rand_set) # 추출한 2개 인덱스가 빠진 6개의 인덱스를 가진 list ex> [0, 1, 3, 5, 6, 7]
        for rand in rand_set: 
            dict_users[i].extend(idxs[rand * num_imgs: (rand + 1) * num_imgs])
    for i in range(n_user):
        x_datasets[i] = np.take(x, dict_users[i], axis=0)
        y_datasets[i] = np.take(labels, dict_users[i], axis=0)
    return x_datasets, y_datasets

def diri_split_data(x_train, y_train, n_user, alpha):
    x, labels = x_train, y_train
    y_train_index = np.argmax(labels, axis=1)
    
    x_datasets = [[] for _ in range(n_user)]
    y_datasets = [[] for _ in range(n_user)]

    t_classes = len(np.unique(y_train_index))

    t_idx_slice = [[] for _ in range(n_user)]

    for k in range(t_classes):
        t_idx_k = np.where(y_train_index == k)[0] #해당 클래스의 index(sorted됨)
        np.random.shuffle(t_idx_k) #해당 클래스의 Random Index

        prop = np.random.dirichlet(np.repeat(alpha, n_user))
        #alpha값을 유저수만큼 복사한 리스트 [0.1, 0.1, 0.1, ..., 0.1]을 dirichlet 함수에 넣는다
        #dirichlet() : 전체합이 1이면서, 길이는 input 리스트의 길이와 같은 리스트를 배출한다.
        #alpha의 값이 클수록 분포가 bias되는 경향이 있다. -> 그래서 default값은 0.1 (편향 분포를 위해서)

        t_prop = (np.cumsum(prop) * len(t_idx_k)).astype(int)[:-1]
        # np.cumsum : 리스트 방향으로 누적 합의 리스트 반출, ex. np.cumsum([[1,3],[5,6]]) = [1 4 9 15]
        # dirichlet 전체합이 1이니까, 리스트의 마지막 값은 t_idx_k의 길이이다.

        t_idx_slice = idx_slicer(t_idx_slice, t_idx_k, t_prop)

    for i in range(n_user):
        np.random.shuffle(t_idx_slice[i])


    for i in range(n_user):
        x_datasets[i] = np.take(x, t_idx_slice[i], axis=0)
        y_datasets[i] = np.take(labels, t_idx_slice[i], axis=0)

    return x_datasets, y_datasets

def idx_slicer(t_idx_slice, idx_k, prop):
    diri_shard = np.split(idx_k, prop)
    #print(f'디리클레 자른 샤드 : {diri_shard}')
    random.shuffle(diri_shard)
    return [idx_j + idx.tolist() for idx_j, idx in zip(t_idx_slice, diri_shard)]
    #idx_k : k 라벨에 속하는 셔플된 인덱스
    #prop : 디리클레 분포로 만들어진 리스트(자를 위치)
    #np.split(idx_k, prop) : 디리클레 분포로 쪼개진 Shard
    #idx_slice : 유저수만큼의 길이의 빈 2차원 배열
        #idx_j : 해당 유저의 idx_slice 위치
        #idx : 해당 유저에 라벨당 하나의 shard 부여

# Random한 개수의 random 데이터
def diff_split_data(x_train, y_train, n_user):
    x_datasets = [[] for _ in range(n_user)]
    y_datasets = [[] for _ in range(n_user)]

    data_idx = [i for i in range(len(x_train))] # [0, 1, 2, ... , 59999]
    np.random.shuffle(data_idx) # [0~59999 셔플된 값]

    slice_index = np.random.choice(data_idx, n_user-1, replace=False) #랜덤으로 자를 위치 선정
    slice_index = np.sort(slice_index) # 선정된 위치 정렬
    slice_index = np.concatenate(([0], slice_index, [len(x_train)-1]))
    print(f'[Dataset 스플릿 인덱스 확인] : {slice_index}')
    for i in range(n_user):
        user_data_index = data_idx[slice_index[i]:slice_index[i+1]] #유저가 할당받을 데이터셋의 인덱스
        x_datasets[i] = np.take(x_train, user_data_index, axis=0)
        y_datasets[i] = np.take(y_train, user_data_index, axis=0)
    
    return x_datasets, y_datasets

#TODO: random한 개수의 Non_iid 데이터 split method인데, 아직 미완성 -> 신경 안써도 될듯?
def ran_non_iid_split_data(x_train, y_train, n_user):
    x, y = x_train, y_train

    tmp_y = np.argmax(y, axis =1)

    x_datasets = [[] for _ in range(n_user)]
    y_datasets = [[] for _ in range(n_user)]

    unique_labels = np.unique(tmp_y, return_counts=False)
    for label in unique_labels:
        label_indices = np.where(tmp_y == label)[0]

