#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 12:47:21 2018

@author: dengyuzhao
"""

import os
import tensorflow as tf
import numpy as np
import test

#mnist=input_data.read_data_sets("/Users/dengyuzhao/Downloads/MNIST_data/",one_hot=True)

#设置参数
numtrain = 200
numtest = 50

BATCH_SIZE=50 #一个训练BATCH中的训练样本数
LEARNING_RATE_BASE=0.1#学习率先设置很大，然后逐渐减小
LEARNING_RATE_DECAY=0.99#学习率的衰减率
REGULARIZATION_RATE=0.0001#正则化参数
TRAINING_STEPS=8000#训练轮数
MOVING_AVERAGE_DECAY=0.99#滑动平均衰减率

#模型保存的路径和文件名
MODEL_SAVE_PATH='/Users/dengyuzhao/Downloads/齐朝晖论文/'
MODEL_NAME='face_train*'


def train():
    x=tf.placeholder(tf.float32,
                     shape=[BATCH_SIZE,test.IMAGE_SIZE,test.IMAGE_SIZE,test.NUM_CHANNELS],name='x_input')
    y_=tf.placeholder(tf.float32,shape=[BATCH_SIZE,test.OUTPUT_NODE],name='y_input')
    
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y=test.inference(x,train,regularizer=regularizer)
    global_step=tf.Variable(0,trainable=False)
    
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection("losses"))
    
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,
                                             global_step,
                                             TRAINING_STEPS/BATCH_SIZE,
                                             LEARNING_RATE_DECAY)
    
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='Train')
        
    #初始化tensorflow持久化类
#    saver=tf.train.Saver()
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
####--------------------------设置BATCH_SIZE------------------------####
        for i in range(TRAINING_STEPS):
            ID = np.random.randint(0,199,BATCH_SIZE)
            xs=test.train_data[ID]
            ys=test.train_label[ID]
            _,loss_value,step=sess.run([train_op, loss, global_step],feed_dict={x:xs,y_:ys})
            
            if i % 50 ==0:
                print("After %d steps training steps,loss on training batch is %g"%(step,loss_value))
#                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
 
    
    ypre = sess.run(y, feed_dict={x: test_data})
    
    y1 = np.argmax(ypre)
    y2 = np.argmax(test_label)

    accuracy = np.sum(y1-y2)
    accuracy = accuracy/numtest
    
    print('Accuracy_learn ============>>  ', accuracy)
    
    

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
        
        
        
    
    
    