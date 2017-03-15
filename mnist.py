# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import tensorflow as tf
import sys
import urllib

if sys.version_info[0] >= 3:
  from urllib.request import urlretrieve
else:
  from urllib import urlretrieve

LOGDIR = '/tmp/mnist_tutorial/'
GITHUB_URL ='https://raw.githubusercontent.com/mamcgrath/TensorBoard-TF-Dev-Summit-Tutorial/master/'

### MNIST EMBEDDINGS ###
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)
### Get a sprite and labels file for the embedding projector ###
urlretrieve(GITHUB_URL + 'labels_1024.tsv', LOGDIR + 'labels_1024.tsv')
urlretrieve(GITHUB_URL + 'sprite_1024.png', LOGDIR + 'sprite_1024.png')

# Add convolution layer
def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Add fully connected layer
def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):
  tf.reset_default_graph()
  sess = tf.Session()

  # Setup placeholders, and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  tf.summary.image('input', x_image, 3)
  y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

  if use_two_conv:
    conv1 = conv_layer(x_image, 1, 32, "conv1")
    conv_out = conv_layer(conv1, 32, 64, "conv2")
  else:
    conv1 = conv_layer(x_image, 1, 64, "conv")
    conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])


  if use_two_fc:
    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
    embedding_input = fc1
    embedding_size = 1024
    logits = fc_layer(fc1, 1024, 10, "fc2")
  else:
    embedding_input = flattened
    embedding_size = 7*7*64
    logits = fc_layer(flattened, 7*7*64, 10, "fc")

  with tf.name_scope("xent"):
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="xent")
    tf.summary.scalar("xent", xent)

  with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

  summ = tf.summary.merge_all()


  embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
  assignment = embedding.assign(embedding_input)
  saver = tf.train.Saver()

  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(LOGDIR + hparam)
  writer.add_graph(sess.graph)

  config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
  embedding_config = config.embeddings.add()
  embedding_config.tensor_name = embedding.name
  embedding_config.sprite.image_path = LOGDIR + 'sprite_1024.png'
  embedding_config.metadata_path = LOGDIR + 'labels_1024.tsv'
  # Specify the width and height of a single thumbnail.
  embedding_config.sprite.single_image_dim.extend([28, 28])
  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

  for i in range(2001):
    batch = mnist.train.next_batch(100)
    if i % 5 == 0:
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
      writer.add_summary(s, i)
    if i % 500 == 0:
      sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
      saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
  conv_param = "conv=2" if use_two_conv else "conv=1"
  fc_param = "fc=2" if use_two_fc else "fc=1"
  return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
  # You can try adding some more learning rates
  for learning_rate in [1E-4]:

    # Include "False" as a value to try different model architectures
    for use_two_fc in [True]:
      for use_two_conv in [True]:
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
        hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
        print('Starting run for %s' % hparam)

	    # Actually run with the new settings
        mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)


if __name__ == '__main__':
  main()
