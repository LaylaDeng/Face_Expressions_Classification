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

CONV1_SIZE=5
CONV1_DEEP=8

CONV2_SIZE=5
CONV2_DEEP=16

FC_SIZE=512

train_data = np.zeros((numtrain,32,32,3))
train_label = np.zeros((numtrain,2))

test_data = np.zeros((numtest, 32,32,3))
test_label = np.zeros((numtest,2))
# 将图像数据存储到data[i]

for i in range(numtrain):
    path = {"path1":"/Users/dengyuzhao/Downloads/齐朝晖论文/pic_data/"+str(i+1)+".jpg", 
            "path2":"/Users/dengyuzhao/Downloads/齐朝晖论文/label.txt"}            
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
    path = {"path1":"/Users/dengyuzhao/Downloads/齐朝晖论文/pic_data/"+str(i+201)+".jpg", 
            "path2":"/Users/dengyuzhao/Downloads/齐朝晖论文/label.txt"}            
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
    
    with tf.variable_scope('layer1-conv1'):
        conv1_weights=tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.tanh(tf.nn.bias_add(conv1,conv1_biases))
        
    with tf.name_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    with tf.variable_scope('layer3-conv2'):
        conv2_weights=tf.get_variable("weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.tanh(tf.nn.bias_add(conv2,conv2_biases))
        
    with tf.name_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    new=tf.reshape(pool2,[pool_shape[0],nodes])
    
    
    with tf.variable_scope('layer5-fc1'):
        fc1_weights=tf.get_variable('weight',shape=[nodes,FC_SIZE],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable('bias',shape=FC_SIZE,
                                   initializer=tf.constant_initializer(0.1))
        
        fc1=tf.nn.tanh(tf.matmul(new,fc1_weights)+fc1_biases)
        ##---如果是训练集，则要dropout使模型更鲁棒-------
        if train:
            fc1=tf.nn.dropout(fc1,0.5)
            
            
    with tf.variable_scope('layer6-fc2'):
        fc2_weights=tf.get_variable('weight',shape=[FC_SIZE,NUM_LABELS],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases=tf.get_variable('biases',shape=[NUM_LABELS],
                                   initializer=tf.constant_initializer(0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases
        
    return logit
        
