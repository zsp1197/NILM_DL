from my_nmt.tf_tools import *
from my_nmt.theiterator import *
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

# embedding
embedding_encoder = tf.get_variable(
    "embedding_encoder", [7709, 128], tf.float32)
embedding_decoder = tf.get_variable(
    "embedding_decoder", [17191, 128], tf.float32)
# encoder
source = tf.transpose(iterator.source)
# Look up embedding, emp_inp: [max_time, batch_size, num_units]
# 分成几个batch
encoder_emb_inp = tf.nn.embedding_lookup(
    embedding_encoder, source)
# Encoder_outpus: [max_time, batch_size, num_units]
cell_encoder = tf.contrib.rnn.MultiRNNCell(
    [tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.), tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.)])
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    cell_encoder,
    encoder_emb_inp,
    dtype=tf.float32,
    sequence_length=iterator.source_sequence_length,
    time_major=True,
    swap_memory=True)
# Decoder
tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant('<s>')),
                     tf.int32)
tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant('</s>')),
                     tf.int32)
cell_decoder = tf.contrib.rnn.MultiRNNCell(
    [tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.), tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.)])
## train
target_input = iterator.target_input
target_input = tf.transpose(target_input)
decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, target_input)
## Helper
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp, iterator.target_sequence_length,
    time_major=True)
my_decoder = tf.contrib.seq2seq.BasicDecoder(
    cell_decoder,
    helper,
    encoder_state, )
# Dynamic decoding
outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
    my_decoder,
    output_time_major=True,
    swap_memory=True)
sample_id = outputs.sample_id
## protection_layer
output_layer = layers_core.Dense(17191, use_bias=False, name="output_projection")
logits = output_layer(outputs.rnn_output)
# loss
target_output = iterator.target_output
target_output = tf.transpose(target_output)
max_time = get_max_time(target_output)
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
# 下面不懂
target_weights = tf.sequence_mask(
    iterator.target_sequence_length, max_time, dtype=logits.dtype)
target_weights = tf.transpose(target_weights)
batch_size = tf.size(iterator.source_sequence_length)
loss = tf.reduce_sum(
    crossent * target_weights) / tf.to_float(batch_size)

# train
word_count = tf.reduce_sum(iterator.source_sequence_length) + tf.reduce_sum(iterator.target_sequence_length)
predict_count = tf.reduce_sum(iterator.target_sequence_length)
global_step = tf.Variable(0, trainable=False)
params = tf.trainable_variables()
# 学习率后面加入动态调整
learning_rate = tf.constant(1.0)
opt = tf.train.AdagradOptimizer(learning_rate)
gradients = tf.gradients(loss, params, colocate_gradients_with_ops=True)
update = opt.apply_gradients(zip(gradients, params), global_step=global_step)
# update=tf.train.AdagradOptimizer(learning_rate).minimize(loss=loss)
