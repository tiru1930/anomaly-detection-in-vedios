from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np 

class DetectionModel(object):
    """docstring for DetectionModel"""
    def __init__(self):
        super(DetectionModel, self).__init__()

    def model(self,inputs,training=True):
        
        net = tf.layers.conv3d(inputs=inputs, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

        net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

        net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

        net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])  

        net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID')
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])  
        net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID')
        net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

        net = tf.layers.flatten(net)
        net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu)
        net = tf.identity(net, name='fc1')
        net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

        net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu)
        net = tf.identity(net, name='fc2')
        net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

        net = tf.layers.dense(inputs=net, units=32, activation=tf.nn.relu)
        net = tf.identity(net, name='fc2')
        net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

        net = tf.layers.dense(inputs=net, units=1, activation=None)
        net = tf.identity(net, name='logits')
        return net


class C3Dmodel(Model):
    """docstring for C3Dmodel"""
    def __init__(self):
        super(C3Dmodel, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(filters=64,kernel_size=3,padding='SAME',activation=tf.keras.layers.ReLU)
        self.conv2 = tf.keras.layers.Conv3D(filters=128,kernel_size=3,padding='SAME',activation=tf.keras.layers.ReLU)
        self.conv3 = tf.keras.layers.Conv3D(filters=256,kernel_size=3,padding='SAME',activation=tf.keras.layers.ReLU)
        self.conv4 = tf.keras.layers.Conv3D(filters=512,kernel_size=3,padding='SAME',activation=tf.keras.layers.ReLU)
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2),padding='SAME')
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=2,strides=2,padding='SAME')
        self.fc1   = tf.keras.layers.Dense(4096,activation=tf.keras.layers.ReLU)
        self.fc2   = tf.keras.layers.Dense(512,activation=tf.keras.layers.ReLU)
        self.fc3   = tf.keras.layers.Dense(32,activation=tf.keras.layers.ReLU)
        self.fc4   = tf.keras.layers.Dense(1)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.flatten = tf.keras.layers.Flatten()

    def call(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x = self.conv4(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        return self.fc4(x)

def main():
    sample = np.random.rand(1,16,240,320,3)
    print(sample.shape)
    model = C3Dmodel()
    print(model(sample))
    
if __name__ == '__main__':
    main()

