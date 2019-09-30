import tensorflow as tf

from tensorflow import ConfigProto

import time, os
import scipy.io as sio
import numpy as np
from random import shuffle

from nets import nets_factory
from dataloader import Dataloader
import op_util

home_path = os.path.dirname(os.path.abspath(__file__))


tf.app.flags.DEFINE_string('dataset', 'cifar100',
                           'Distillation method : cifar100, TinyImageNet, CUB200')
tf.app.flags.DEFINE_integer('hintLayerIndex', 16)
tf.app.flags.DEFINE_integer('guidedLayerIndex',2)

def main(_):
    ### define path and hyper-parameter
    model_name   = 'ResNet'
    val_batch_size = 200
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_num = '0'
    weight_decay = 5e-4
    train_images, train_labels, val_images, val_labels, pre_processing, teacher = Dataloader("cifar100", home_path)
    num_label = int(np.max(train_labels)+1)

    dataset_len, *image_size = train_images.shape

    with tf.Graph().as_default() as graph:

        # make placeholder for inputs
        image_ph = tf.placeholder(tf.uint8, [None]+image_size)
        label_ph = tf.placeholder(tf.int32, [None])
        is_training_ph = tf.placeholder(tf.bool,[])

        # pre-processing
        image = pre_processing(image_ph, is_training_ph)
        label = tf.contrib.layers.one_hot_encoding(label_ph, num_label, on_value=1.0)

        ## load Net
        class_loss, accuracy = MODEL(model_name, "Teacher", weight_decay, image, label,
                                     is_training_ph, reuse = False, drop = True, Distillation = None)
        ## make placeholder and summary op for training and validation results
        train_acc_place = tf.placeholder(dtype=tf.float32)
        val_acc_place   = tf.placeholder(dtype=tf.float32)
        val_summary = [tf.summary.scalar('accuracy/training_accuracy',   train_acc_place),
                       tf.summary.scalar('accuracy/validation_accuracy', val_acc_place)]
        val_summary_op = tf.summary.merge(list(val_summary), name='val_summary_op')

        ## start training
        config = ConfigProto()
        config.gpu_options.visible_device_list = gpu_num
        config.gpu_options.allow_growth=True

        val_itr = len(val_labels)//val_batch_size

        with tf.Session(config=config) as sess:

            saver = tf.train.import_meta_graph('teacherV.ckpt.meta')
            saver.restore(sess,'teacherV.ckpt')
            print ('%d Teacher params assigned')
            idx = list(range(train_labels.shape[0]))
            shuffle(idx)
            sum_val_accuracy = []
            sess.run(tf.global_variables_initializer())
            for i in range(val_itr):
                val_batch = val_images[i*val_batch_size:(i+1)*val_batch_size]
                acc = sess.run(accuracy, feed_dict = {image_ph : val_batch,
                                                      label_ph : np.squeeze(val_labels[i*val_batch_size:(i+1)*val_batch_size]),
                                                      is_training_ph : False})
                sum_val_accuracy.append(acc)
    sum_val_accuracy= np.mean(sum_val_accuracy)*100
    print(sum_val_accuracy)

def MODEL(model_name, scope, weight_decay, image, label, is_training, reuse, drop, Distillation):
    network_fn = nets_factory.get_network_fn(model_name, weight_decay = weight_decay)
    end_points = network_fn(image, label, scope, is_training=is_training, reuse=reuse, drop = drop, Distill=Distillation,hintLayerIndex=FLAGS.hintLayerIndex,guidedLayerIndex=FLAGS.guidedLayerIndex)

    loss = tf.losses.softmax_cross_entropy(label,end_points['Logits'])
    if Distillation == 'DML':
        tf.add_to_collection('teacher_class_loss',tf.losses.softmax_cross_entropy(label,end_points['Logits_tch']))
    accuracy = tf.contrib.metrics.accuracy(tf.to_int32(tf.argmax(end_points['Logits'], 1)), tf.to_int32(tf.argmax(label, 1)))
    return loss, accuracy

def learning_rate_scheduler(Learning_rate, epochs, decay_point, decay_rate):
    with tf.variable_scope('learning_rate_scheduler'):
        e, ie, te = epochs
        for i, dp in enumerate(decay_point):
            Learning_rate = tf.cond(tf.greater_equal(e, ie + int(te*dp)), lambda : Learning_rate*decay_rate,
                                    lambda : Learning_rate)
        tf.summary.scalar('learning_rate', Learning_rate)
        return Learning_rate


def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                           comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exist")

    if not output_node_names:
        print("You need to supply the name of the output node")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(args.meta_graph_path, clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )
    return frozen_graph

if __name__ == '__main__':
    tf.app.run()


