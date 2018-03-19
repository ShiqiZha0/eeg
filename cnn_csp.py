# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:00:12 2017

@author: zhaos
"""


import csv
import struct
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt


# MNIST data is stored in binary format, 
# and we transform them into numpy ndarray objects by the following two utility functions
         
def read_image(file_name,images):

    f = csv.reader(open(file_name, encoding='utf-8'))               
    rows = 240
    columns = 2
    data_array = []
    for row in f:
        for cell in row:
            data_array.append(cell)
    data_array = np.array(data_array).reshape(images, rows, columns)    

    return data_array
 
def read_label(file_name):
    f = csv.reader(open(file_name, encoding='utf-8'))
    label_array = []
    for row in f:
        for cell in row:
            label_array.append(int(cell))
    return label_array
 
print ("Start processing MNIST handwritten digits data...")
#train_x_data = read_image("MNIST_data/train-images-idx3-ubyte.gz")  # shape: 2x12x2

train_x_data = read_image("data/k3b/train.csv",60)
train_y_data = read_label("data/k3b/train_label.csv")   
test_x_data = read_image("data/k3b/test.csv",30) 
test_y_data = read_label("data/k3b/test_label.csv")



def csp_data(x_data,y_data,num):
    a = np.zeros([2,num,240,2])
    sample_x, sample_y = 0,0
    for i in range(len(y_data)):
        if y_data[i] == 0:
            for col in range(240):
                for row in range(2):
                    a[0][sample_x][col][row] = x_data[sample_x + sample_y][col][row] 
            sample_x += 1
        else:
            for col in range(240):
                for row in range(2):
                    a[1][sample_y][col][row] = x_data[sample_x + sample_y][col][row]
            sample_y += 1
    return a
#print (b)
train_csp_raw = csp_data(train_x_data,train_y_data,30)
test_csp_raw = csp_data(test_x_data,test_y_data,15)
train_csp = csp.CSP(train_csp_raw)
test_csp = csp.CSP(test_csp_raw)

train_x_data = train_csp
test_x_data = test_csp    

train_x_data = train_x_data.reshape(train_x_data.shape[0], train_x_data.shape[1], train_x_data.shape[2], 1).astype(np.float32)
test_x_data = test_x_data.reshape(test_x_data.shape[0], test_x_data.shape[1], test_x_data.shape[2], 1).astype(np.float32)
train_x_minmax = train_x_data 
test_x_minmax = test_x_data


lb = preprocessing.LabelBinarizer()
lb.fit(train_y_data)
train_y_data_trans = lb.transform(train_y_data)
test_y_data_trans = lb.transform(test_y_data)
train_zero = np.zeros(60)
test_zero = np.zeros(30)
#print (train_y_data_trans)
train_y_data_trans = np.column_stack([train_y_data_trans,train_zero]).astype(int)
test_y_data_trans = np.column_stack([test_y_data_trans,test_zero]).astype(int)
for i in train_y_data_trans:
    if i[0] == 0:
        i[1] = 1
for j in test_y_data_trans:
    if j[0] == 0:
        j[1] = 1


#print (train_y_data_trans)
#print (train_y_data_trans.shape)
 
print ("Start evaluating CNN model by tensorflow...")
 

x = tf.placeholder(tf.float32, shape=[None, 240, 2, 1])
y_ = tf.placeholder(tf.float32, [None, 2])

# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
#def conv2d(x, W):
    # `tf.nn.conv2d()` computes a 2-D convolution given 4-D `input` and `filter` tensors
    # input tensor shape `[batch, in_height, in_width, in_channels]`, batch is number of observation 
    # filter tensor shape `[filter_height, filter_width, in_channels, out_channels]`
    # strides: the stride of the sliding window for each dimension of input.
    # padding: 'SAME' or 'VALID', determine the type of padding algorithm to use
    #return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# First convolutional layer
# Convolution: compute 32 features for each 5x5 patch
# Max pooling: reduce image size to 14x14.

W_conv1 = weight_variable([4, 1, 1, 16])
b_conv1 = bias_variable([16])
#h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_conv1 = tf.nn.relu(tf.nn.conv2d(x,  W_conv1, strides=[1,4,1,1], padding='VALID') + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
# Convolution: compute 64 features for each 5x5 patch
# Max pooling: reduce image size to 7x7
W_conv2 = weight_variable([1, 2, 16, 32])
b_conv2 = bias_variable([32])

#h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1,  W_conv2, strides=[1,1,2,1], padding='VALID') + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)
print (h_conv2.shape)
# Densely connected layer
# Fully-conected layer with 1024 neurons
W_fc1 = weight_variable([32*60*1, 256])
b_fc1 = bias_variable([256])

print (h_conv2.shape)
#h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 41 * 64])
h_flat = tf.reshape(h_conv2, [-1, 32*60*1])



h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

# Dropout
# To reduce overfitting, we apply dropout before the readout layer.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([256, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #matmul xiangcheng

regularizer = tf.contrib.layers.l2_regularizer(0.01)
regularization = regularizer(W_fc1) + regularizer(W_fc2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)) + regularization
# Train and evaluate

y = tf.argmax(y_conv,1)
# loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(1e-4)
# optimizer = tf.train.GradientDescentOptimizer(1e-4)
train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

 
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(100):
    #f1 = open('test.txt','w')
    #f1.write(str(sess.run(b_fc1)))
    #print (train_x_minmax.shape)
    print (b_fc1.eval(session=sess))
    sample_index = np.random.choice(train_x_minmax.shape[0], 5)
   # print (sample_index)
    batch_xs = train_x_minmax[sample_index, :]
   # sample_index_y = np.random.choice()
    #batch_ys = train_y_data_trans
    batch_ys = train_y_data_trans[sample_index, :]
    if step % 2 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print ("step %d, training accuracy %g" % (step, train_accuracy))
        print ("test accuracy %g" % sess.run(accuracy, feed_dict={
    x: test_x_minmax, y_: test_y_data_trans, keep_prob: 1.0}))
        print (sess.run(y, feed_dict={
    x: test_x_minmax, y_: test_y_data_trans, keep_prob: 1.0}))        
        #print (batch_xs)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
 
print ("test accuracy %g" % sess.run(accuracy, feed_dict={
    x: test_x_minmax, y_: test_y_data_trans, keep_prob: 1.0}))
game_signal = (sess.run(y, feed_dict={
    x: test_x_minmax, y_: test_y_data_trans, keep_prob: 1.0}))


'''
import pygame
pygame.init()
from pygame.color import THECOLORS
import sys
    

def loadcar(xloc,yloc):
    my_car=pygame.image.load('ok1.jpg')
    locationxy=[xloc,yloc]
    screen.blit(my_car,locationxy)
    pygame.display.flip()

    
pygame.init()
screen=pygame.display.set_caption('eeg_demo')
screen=pygame.display.set_mode([640,700])
screen.fill([255,255,255])
#lineleft()
#lineright()
#linemiddle()

while True: # main game loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    xloc = 260
    for looper in range(700,-200,-30):
       # for i in range(len(game_signal)):
       pygame.time.delay(200)
       if game_signal[int((700-looper)/30)] == 1:
           pygame.draw.rect(screen,[255,255,255],[xloc,(looper+132),83,132],0)
           xloc = xloc - 20
           loadcar(xloc,looper)
       else:
           pygame.draw.rect(screen,[255,255,255],[xloc,(looper+132),83,132],0)
           xloc = xloc + 20
           loadcar(xloc,looper)           
         
exit()
'''
