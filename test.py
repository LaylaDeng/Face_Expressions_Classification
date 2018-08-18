import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import os
import time


#log 日志级别设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #只显示Warnings和Errors
tf.reset_default_graph()

### ------  Paramaters -------------

numtrain = 200
numtest = 50

IMPUT_NODE=3072
OUTPUT_NODE=2

IMAGE_SIZE=32
NUM_CHANNELS=3
NUM_LABELS=2


train_data = np.zeros((numtrain,32,32,3))
train_label = np.zeros((numtrain,2))

test_data = np.zeros((numtest, 32,32,3))
test_label = np.zeros((numtest,2))
# 将图像数据存储到data[i]

for i in range(numtrain):
    path = {"path1":"/Users/dengyuzhao/Downloads/mytry_face-detect/pic_data/"+str(i+1)+".jpg",
            "path2":"/Users/dengyuzhao/Downloads/mytry_face-detect/label.txt"}
    f1 = cv2.imread(path["path1"])
    train_data[i] = f1


f2 = open(path["path2"])
num=0
for line in f2:
    list = line.strip('\n').split(' ')
    if list[0]=='%d'%(numtrain+1):
        break
    train_label[num][:]= list[1:3]
    num+=1
f2.close()


for i in range(numtest):
    path = {"path1":"/Users/dengyuzhao/Downloads/mytry_face-detect/pic_data/"+str(i+201)+".jpg",
            "path2":"/Users/dengyuzhao/Downloads/mytry_face-detect/label.txt"}            
    f1 = cv2.imread(path["path1"])
    test_data[i] = f1


f2 = open(path["path2"])
num=0
for line in f2:
    list = line.strip('\n').split(' ')
    if int(list[0])<201:
        continue
    if int(list[0])>250:
        break
    test_label[num][:]= list[1:3]
    num+=1
f2.close()


###--------------前向传播过程----------------------------


def inference(input_tensor,train,regularizer):
    
    conv1 = tf.layers.conv2d(inputs = input_tensor, filters = 5, kernel_size = 5,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            padding = "same", activation = tf.nn.tanh)
    pool1  = tf.layers.max_pooling2d(inputs=conv1, pool_size = [2,3], strides = [2,3])
    
    conv2 = tf.layers.conv2d(inputs = pool1, filters = 15, kernel_size = 5,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             padding = "same", activation = tf.nn.tanh)
    pool2  = tf.layers.max_pooling2d(inputs=conv2, pool_size = [2,3], strides = [2,3])
    
    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    pool2flat=tf.reshape(pool2,[pool_shape[0],nodes])
 
    layer5 = tf.layers.dense(inputs= pool2flat, units = 1024, activation=tf.nn.tanh)
        

    if train:
        layer5 = tf.layers.dropout(inputs = layer5, rate = 0.5)
    
    fc2_weights=tf.get_variable("weight",[1024,2],initializer=tf.truncated_normal_initializer(stddev=0.1))
    bias=tf.get_variable("biases",shape=[2],initializer=tf.constant_initializer(0.1))
    logits = tf.matmul(layer5,fc2_weights)+bias
    if regularizer!=None:
                tf.add_to_collection('losses',regularizer(fc2_weights))
    output = tf.nn.softmax(logits)
    return output
