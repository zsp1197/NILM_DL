# Created by zhai at 2018/1/28
# Email: zsp1197@163.com
import tensorflow as tf

num_of_appliance = 12
cell_size = 15
x = tf.placeholder(dtype=tf.float32, shape=[None, cell_size, 5])
y = tf.placeholder(dtype=tf.float32, shape=[None, cell_size, num_of_appliance])

# weight=tf.get_variable(name='input_weight',shape=[])
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=5, forget_bias=1.0)
cell_outputs, cell_states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, time_major=False,dtype=tf.float32)
outputWeight=tf.Variable(tf.random_normal([5, num_of_appliance]))
outputBias=tf.Variable(tf.random_normal([num_of_appliance]))

output=tf.matmul(cell_outputs,outputWeight)+outputBias
