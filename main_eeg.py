# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 00:00:10 2018

@author: zhaos
"""
import csv
import tensorflow as tf
import importlib
import tensorflow.python.platform
import os
import numpy as np
from progress.bar import Bar
from datetime import datetime
from tensorflow.python.platform import gfile
from data import *
from evaluate import evaluate
from sklearn import preprocessing
import BNN_cifar10
import csp
import sys
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
import nnUtils
from nnUtils import *


#sys.path.append('c:\Users\zhaos\Desktop\')
model_path = 'c:/Users/zhaos/Desktop/' + 'eeg.ckpt'

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
print ("Start processing eeg data...")




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


train_x_data = train_x_data.reshape(train_x_data.shape[0], 480, 1, 1).astype(np.float32)
test_x_data = test_x_data.reshape(test_x_data.shape[0], 480, 1, 1).astype(np.float32)

#train_x_data = train_x_data.reshape(train_x_data.shape[0], 480, 1, 1).astype(np.float32)
#test_x_data = test_x_data.reshape(test_x_data.shape[0], 480, 1, 1).astype(np.float32)

#train_x_data = train_x_data.reshape(train_x_data.shape[0], train_x_data.shape[1], train_x_data.shape[2], 1).astype(np.float32)
#test_x_data = test_x_data.reshape(test_x_data.shape[0], test_x_data.shape[1], test_x_data.shape[2], 1).astype(np.float32)
lb = preprocessing.LabelBinarizer()
lb.fit(train_y_data)
train_y_data_trans = lb.transform(train_y_data)
test_y_data_trans = lb.transform(test_y_data)
train_zero = np.zeros(60)
test_zero = np.zeros(30)
train_y_data_trans = np.column_stack([train_y_data_trans,train_zero]).astype(int)
test_y_data_trans = np.column_stack([test_y_data_trans,test_zero]).astype(int)
for i in train_y_data_trans:
    if i[0] == 0:
        i[1] = 1
for j in test_y_data_trans:
    if j[0] == 0:
        j[1] = 1
x = tf.placeholder(tf.float32, shape=[None, 480, 1, 1])
y_ = tf.placeholder(tf.float32, [None, 2])

#m = importlib.import_module('models.BNN_cifar10')
#model = m.model
#y = BNN_cifar10.model(x, is_training=True)


def binarize(x):
    g = tf.get_default_graph()
    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            x=tf.clip_by_value(x,-1,1)
            return tf.sign(x)
        
def binarize_0(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            x=tf.clip_by_value(x,0,1)
            return tf.sign(x)

def BinarizedSpatialConvolution(x,nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='BinarizedSpatialConvolution'):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_op_scope([x], None, name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bin_w = binarize(w)
            bin_x = binarize_0(x)
            out = tf.nn.conv2d(bin_x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out,bin_w,bin_x

def BinarizedWeightOnlySpatialConvolution(x,nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='BinarizedWeightOnlySpatialConvolution'):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_op_scope([x], None, name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bin_w = binarize(w)
            out = tf.nn.conv2d(x, bin_w, strides=[1, dH, dW, 1], padding=padding)            
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
        
def BinarizedAffine(x,nInputPlane,nOutputPlane, bias=True, name=None, reuse=None):
        with tf.variable_op_scope([x], name, 'Affine', reuse=reuse):
            bin_x = binarize_0(x)
            reshaped = tf.reshape(x, [-1, nInputPlane])
            #reshaped = tf.reshape(bin_x, [x.get_shape().as_list()[0], -1])
           # nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            bin_w = binarize(w)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output   


def bn(x, is_training):  
    x_shape = x.get_shape()  
    params_shape = x_shape[-1:]  
  
    axis = list(range(len(x_shape) - 1))  
  
    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())  
    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())  
  
    moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)  
    moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)  
  
    # These ops will only be preformed when training.  
    mean, variance = tf.nn.moments(x, axis)  
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)  
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)  
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)  
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)  
  
    mean, variance = control_flow_ops.cond(  
        is_training, lambda: (mean, variance),  
        lambda: (moving_mean, moving_variance))  
  
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)  
        
def HardTanh(x,name='HardTanh'):
        with tf.variable_op_scope([x], None, name):
            return tf.clip_by_value(x,-1,1)

def wrapNN(x,f,*args,**kwargs):
        return f(x,*args,**kwargs)

def BatchNormalization(x,*args, **kwargs):
    return tf.contrib.layers.batch_norm(x,*args,**kwargs)

y_1 = BinarizedWeightOnlySpatialConvolution(x,16,1,6,1,6, padding='VALID', bias=False)
y_2 = BatchNormalization(y_1)
y_3 = HardTanh(y_2)
y_4,bin_w,bin_x = BinarizedSpatialConvolution(y_3,32,1,4,1,4, padding='VALID', bias=False)
y_5 = HardTanh(y_4)
bin_y5 = binarize_0(y_5)
y_6 = BinarizedAffine(y_5,640,100, bias=False)
y_7 = BatchNormalization(y_6)
y_8 = HardTanh(y_7)
y_9 = BinarizedAffine(y_8,100,2, bias=False)
y_10 = BatchNormalization(y_9)



#y_3 = HardTanh(y_2)

keep_prob = tf.placeholder(tf.float32)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_10, labels=y_))

y_predict = tf.argmax(y_10,1)
optimizer = tf.train.AdamOptimizer(1e-2)
train = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_10, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



init = tf.initialize_all_variables()
sess = tf.Session()

saver = tf.train.Saver()


with tf.Session() as sess:

    sess.run(init)
    for step in range(100):
        sample_index = np.random.choice(train_x_data.shape[0], 2)
        batch_xs = train_x_data[sample_index, :]
        batch_ys = train_y_data_trans[sample_index, :]
       # print (len(bin_w.eval()))
       # bin_w1 = open('test.txt','w')

       # for a in range(len(bin_w.eval())):
        #    for b in range(len(bin_w.eval()[0])):
         #       for c in range(len(bin_w.eval()[0][0])):
          #          for d in range(len(bin_w.eval()[0][0][0])):
           #             print (1111)
            #            bin_w1.write(str(sess.run(bin_w[a][b][c][d])))

        if step % 2 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                    x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print ("step %d, training accuracy %g" % (step, train_accuracy))
            #print ("test accuracy %g" % sess.run(accuracy, feed_dict={
             #       x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}))

            #print (sess.run(y_predict, feed_dict={
             #       x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}))
            #print (sess.run(bin_x, feed_dict={
             #       x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}))
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        
    print ("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}))

    w_1 = open('w_1.txt','w')
    for element in sess.run(bin_w, feed_dict={
        x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}).flat:
        w_1.write(str(element).strip(".0") + "\n")
    print (sess.run(bin_w, feed_dict={
        x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}).shape)
    print (sess.run(bin_x, feed_dict={
        x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}).shape)    
    x_1 = open('x_1.txt','w')
    for element in sess.run(bin_x, feed_dict={
        x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}).flat:
        x_1.write(str(element).rstrip("0").strip("\.") + "\n")
    y5_bin = open('bin_y5.txt','w')
    for element in sess.run(bin_y5, feed_dict={
        x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}).flat:
        y5_bin.write(str(element).rstrip("0").strip("\.") + "\n")
    print ("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}))
    print ("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}))
    y_4out = open('y_4.txt','w')
    for element in sess.run(y_4, feed_dict={
            x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}).flat:
            y_4out.write(str(element).rstrip("0").strip("\.") + "\n")  
            

    
    
    
    save_path = saver.save(sess, model_path)
    #print (sess.run(bin_x,feed_dict={
     #               x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}))
   # reader = tf.train.NewCheckpointReader(model_path)
    #var_to_shape_map = reader.get_variable_to_shape_map()  
    #for key in var_to_shape_map:  
     #   print("tensor_name: ", key)  
        #print(reader.get_tensor(key))
   # w1 = reader.get_tensor("BatchNorm_35/beta")
    #x = reader.get_tensor("binarize/weight")
    #print (bin_w.shape)
    #print (bin_x.shape)
    #print (nnUtils.binarize(w1).eval())
    #print (type(w1))

with tf.Session() as sess:
  # Restore variables from disk.
      saver.restore(sess, model_path)
      print ("test accuracy %g" % sess.run(accuracy, feed_dict={
             x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}))
      print (sess.run(y_4, feed_dict={
             x: test_x_data, y_: test_y_data_trans, keep_prob: 1.0}))
      print("Model restored.")

    









'''

def count_params(var_list):
    num = 0
    for var in var_list:
        if var is not None:
            num += var.get_shape().num_elements()
    return num


def add_summaries(scalar_list=[], activation_list=[], var_list=[], grad_list=[]):

    for var in scalar_list:
        if var is not None:
            tf.summary.scalar(var.op.name, var)

    for grad, var in grad_list:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    for var in var_list:
        if var is not None:
            tf.summary.histogram(var.op.name, var)
            sz = var.get_shape().as_list()
            if len(sz) == 4 and sz[2] == 3:
                kernels = tf.transpose(var, [3, 0, 1, 2])
                tf.summary.image(var.op.name + '/kernels',
                                 group_batch_images(kernels), max_outputs=1)
    for activation in activation_list:
        if activation is not None:
            tf.summary.histogram(activation.op.name +
                                 '/activations', activation)
            #tf.summary.scalar(activation.op.name + '/sparsity', tf.nn.zero_fraction(activation))


def _learning_rate_decay_fn(learning_rate, global_step):
  return tf.train.exponential_decay(
      learning_rate,
      global_step,
      decay_steps=1000,
      decay_rate=0.9,
      staircase=True)

learning_rate_decay_fn = _learning_rate_decay_fn

def train(model, data,
          batch_size=128,
          learning_rate=1e-2,
          log_dir='./log',
          checkpoint_dir='./checkpoint',
          num_epochs=-1):

    # tf Graph input

    with tf.device('/cpu:0'):
        with tf.name_scope('data'):
            x, yt = data.generate_batches(batch_size)

        #global_step =  tf.get_variable('global_step', shape=[], dtype=tf.int64,
        #                     initializer=tf.constant_initializer(0),
        #                     trainable=False)
        global_step = tf.Variable(tf.zeros(shape=[]),name='global_step',trainable=False)
   # if FLAGS.gpu:
       # device_str='/gpu:' + str(FLAGS.device)
    #else:
        #device_str='/cpu:0'

    device_str='/cpu:0'
    print (x)
    with tf.device(device_str):
        y = model(x, is_training=True)
        
        # Define loss and optimizer

        with tf.name_scope('objective'):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y))
            accuracy = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(y, yt, 1), tf.float32))
        opt = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, 'Adam',
                                              gradient_noise_scale=None, gradient_multipliers=None,
                                              clip_gradients=None, #moving_average_decay=0.9,
                                              learning_rate_decay_fn=learning_rate_decay_fn, update_ops=None, variables=None, name=None)
        #grads = opt.compute_gradients(loss)
        #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # loss_avg

    ema = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step, name='average')
    ema_op = ema.apply([loss, accuracy] + tf.trainable_variables())
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

    loss_avg = ema.average(loss)
    tf.summary.scalar('loss/training', loss_avg)
    accuracy_avg = ema.average(accuracy)
    tf.summary.scalar('accuracy/training', accuracy_avg)

    check_loss = tf.check_numerics(loss, 'model diverged: loss->nan')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, check_loss)
    updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([opt]):
        train_op = tf.group(*updates_collection)

    if FLAGS.summary:
        add_summaries( scalar_list=[accuracy, accuracy_avg, loss, loss_avg],
            activation_list=tf.get_collection(tf.GraphKeys.ACTIVATIONS),
            var_list=tf.trainable_variables())
            # grad_list=grads)

    summary_op = tf.summary.merge_all()

    # Configure options for session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(
        config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=gpu_options,
        )
    )
    saver = tf.train.Saver(max_to_keep=5)

    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    num_batches = data.size[0] / batch_size
    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    epoch = 0

    print('num of trainable paramaters: %d' %
          count_params(tf.trainable_variables()))
    while epoch != num_epochs:
        epoch += 1
        curr_step = 0
        # Initializing the variables

        #with tf.Session() as session:
        #    print(session.run(ww))

        print('Started epoch %d' % epoch)
        bar = Bar('Training', max=num_batches,
                  suffix='%(percent)d%% eta: %(eta)ds')
        while curr_step < data.size[0]:
            _, loss_val = sess.run([train_op, loss])
            curr_step += FLAGS.batch_size
            bar.next()

        step, acc_value, loss_value, summary = sess.run(
            [global_step, accuracy_avg, loss_avg, summary_op])
        saver.save(sess, save_path=checkpoint_dir +
                   '/model.ckpt', global_step=global_step)
        bar.finish()
        print('Finished epoch %d' % epoch)
        print('Training Accuracy: %.3f' % acc_value)
        print('Training Loss: %.3f' % loss_value)

        test_acc, test_loss = evaluate(model, FLAGS.dataset,
                                       batch_size=batch_size,
                                       checkpoint_dir=checkpoint_dir)  # ,
        # log_dir=log_dir)
        print('Test Accuracy: %.3f' % test_acc)
        print('Test Loss: %.3f' % test_loss)

        summary_out = tf.Summary()
        summary_out.ParseFromString(summary)
        summary_out.value.add(tag='accuracy/test', simple_value=test_acc)
        summary_out.value.add(tag='loss/test', simple_value=test_loss)
        summary_writer.add_summary(summary_out, step)
        summary_writer.flush()

    # When done, ask the threads to stop.
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    summary_writer.close()


def main(argv=None):  # pylint: disable=unused-argument
    if not gfile.Exists(checkpoint_dir):
        # gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        gfile.MakeDirs(checkpoint_dir)
        model_file = os.path.join('models','BNN_cifar10' + '.py')
        assert gfile.Exists(model_file), 'no model file named: ' + model_file
        gfile.Copy(model_file, checkpoint_dir + '/model.py')
    m = importlib.import_module('.' + FLAGS.model, 'models')
    data = get_data_provider('cifar10', training=True)

    train('BNN_cifar10', data,
          batch_size=128,
          checkpoint_dir=checkpoint_dir,
          log_dir=FLAGS.log_dir,
          num_epochs=-1)

if __name__ == '__main__':
    tf.app.run()
'''