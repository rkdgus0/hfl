import tensorflow as tf
import keras
from keras import layers
#from tensorflow.keras.applications.densenet import DenseNet121
#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications import VGG16, ResNet101
from tensorflow.python.keras import Model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
#from tensorflow.keras.layers import BatchNormalization
from keras.layers import *


#모델 클래스 선언
class McMahanTwoNN(Model):
    '''
    A simple multilayer-perceptron with 2-hidden layers with 200 units
    each using ReLu activations (199,210 total parameters)
    reference] McMahan, Brendan, et al. "Communication-efficient learning of
               deep networks from decentralized data," PMLR, 2017   
    '''
    def __init__(self, input_shape):
        super(McMahanTwoNN, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        self.flatten = Flatten(input_shape=(32, 32, 3))
        self.dense1 = Dense(64, activation='relu', kernel_initializer=initializer)
        self.dense2 = Dense(10)
        self.dense3 = Dense(10, activation='softmax')

        self.build((None,) + tuple(input_shape))

    def call(self, x, training=None, mask=None) -> Model:
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return self.dense3(x)


class McMahanCNN(Model):
    '''
    A CNN with two 5x5 convolution layers (the first with 32 channels,
    the second with 64, each followed with 2x2 max pooling),
    a fully connected layer with 512 units and ReLu activation,
    and a final softmax output layer (1,663,370 total parameters)
    reference] McMahan, Brendan, et al. "Communication-efficient learning of
               deep networks from decentralized data," PMLR, 2017
    '''
    def __init__(self, input_shape):
        super(McMahanCNN, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        self.conv1 = Conv2D(32, 5, activation='relu', padding='same', kernel_initializer=initializer)
        self.pool1 = MaxPool2D(2, strides=2)
        self.conv2 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer=initializer)
        self.pool2 = MaxPool2D(2, strides=2)
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu', kernel_initializer=initializer)
        self.dense2 = Dense(10)
        self.dense3 = Dense(10, activation='softmax')

        self.build((None,) + input_shape)

    def call(self, x, training=None, mask=None) -> Model:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return self.dense3(x)

def define_model(model_name, pre_trained):
    #TODO args에 따라 model 형태 추가해야 함.

    if pre_trained:
        weight = 'imagenet'
    else:
        weight = None

    num_classes = 10
    input_shape = (32, 32, 3)


    if model_name == "mcmahan2NN":
        Model = McMahanTwoNN(input_shape=input_shape)

    elif model_name =="mcmahanCNN":
        Model = McMahanCNN(input_shape=input_shape)

    elif model_name == "VGG16":
        model = VGG16(weights=weight, include_top=False, input_shape=input_shape)
        Model.add(model)
        Model.add(Flatten())
        Model.add(Dense(4096, activation='relu'))
        Model.add(Dropout(0.5))
        Model.add(Dense(2048, activation='relu'))
        Model.add(Dropout(0.5))
        Model.add(Dense(1024, activation='relu'))
        Model.add(Dropout(0.5))
        Model.add(Dense(num_classes, activation='softmax'))
    
    elif model_name in ['resnet50', 'resnet101', 'densenet121']:
        Model = Sequential()
        Model.add(InputLayer(input_shape=(32,32,3)))
        Model.add(Normalization(mean=[0.4914, 0.4822, 0.4465], variance=[0.2023, 0.1994, 0.2010]))
        if model_name == 'resnet50':
            model = ResNet50(weights=weight, include_top=False, input_shape=input_shape)
        elif model_name == 'resnet101':
            model = ResNet101(weights=weight, include_top=False, input_shape=input_shape)
        elif model_name == 'densenet121':
            model = DenseNet121(weights=weight, include_top=False, input_shape=input_shape)
        Model.add(model)
        Model.add(GlobalAveragePooling2D())
        Model.add(Dense(num_classes, activation='softmax'))
    
    else:
        raise ValueError("check model name")
    
    
    return Model



