from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import Multiple



def ResNet_arg_scope(weight_decay=0.0005):
    with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                            mode='FAN_OUT'),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                        biases_initializer=None, activation_fn=None,
                                        ):
        with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                            scale=True, center=True, activation_fn=tf.nn.relu, decay=0.9, epsilon=1e-5,
                                            param_regularizers={'gamma': tf.contrib.layers.l2_regularizer(weight_decay),
                                                                'beta': tf.contrib.layers.l2_regularizer(weight_decay)},
                                            variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                                   'BN_collection']) as arg_sc:
            return arg_sc




def NetworkBlock(x, nb_layers, depth, stride, is_training=False, reuse=False, name='', stdScope=True):
    # with tf.variable_scope(name):
    j = 0;
    for i in range(nb_layers):
        '''x = ResBlock(x, depth, stride = stride if i == 0 else 1,
                         get_feat = True if i == nb_layers-1 else False,
                         is_training = is_training, reuse = reuse, name = 'BasicBlock%d'%i)'''
        strd = stride if i == 0 else 1
        out = tf.contrib.layers.conv2d(x, depth, [3, 3], stride=strd, scope=name + 'conv0%d' % i, trainable=True,
                                       reuse=reuse)
        out = tf.contrib.layers.batch_norm(out, scope=name + 'bn0%d' % i, trainable=True, is_training=is_training,
                                           reuse=reuse)
        #if stdScope==True:
        tf.add_to_collection('feat', out)
        print(out)
        #else:
        #    tf.add_to_collection('featT', out)
        print(stdScope)

        out = tf.contrib.layers.conv2d(out, depth, [3, 3], 1, scope=name + 'conv1%d' % i, trainable=True, reuse=reuse)
        out = tf.contrib.layers.batch_norm(out, scope=name + 'bn1%d' % i, trainable=True, is_training=is_training,
                                           reuse=reuse,
                                           activation_fn=None)
        #if stdScope==True:
        tf.add_to_collection('feat', out)
        print(out)
        #else:
         #   tf.add_to_collection('featT', out)

        if strd > 1 or depth != x.get_shape().as_list()[-1]:
            x = tf.contrib.layers.conv2d(x, depth, [1, 1], stride, scope=name + 'conv2%d' % i, trainable=True,
                                         reuse=reuse)
            x = tf.contrib.layers.batch_norm(x, scope=name + 'bn2%d' % i, trainable=True, is_training=is_training,
                                             reuse=reuse,
                                             activation_fn=None)
        out = x + out
        if i == nb_layers - 1:
            tf.add_to_collection('feat_noact', out)
        out = tf.nn.relu(out)
        #if stdScope==True:
        tf.add_to_collection('feat', out)
        print(out)
        #else:
         #   tf.add_to_collection('featT', out)
    return out


def ResNet(image, label, scope, is_training, reuse=False, drop=False, Distill=None,hintLayerIndex=3,guidedLayerIndex=3):
    end_points = {}

    nChannels = [32, 64, 128, 256] if scope != 'teacher' else [32, 64, 128, 256]

    stride = [1, 2, 2] if scope != 'teacher' else [1, 2, 2]

    n = 1 if scope != 'Teacher' else 5
    with tf.variable_scope(scope):
        std = tf.contrib.layers.conv2d(image, nChannels[0], [3, 3], 1, scope='conv0', trainable=True, reuse=reuse)
        std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable=True, is_training=is_training, reuse=reuse)
        if scope!='Teacher':stdScopy=False
        else : stdScopy=True
        for i in range(len(stride)):
            std = NetworkBlock(std, n, nChannels[1 + i], stride[i], is_training=is_training, reuse=reuse,
                               name='N%d' % i, stdScope=True)
        fc = tf.reduce_mean(std, [1, 2])
        logits = tf.contrib.layers.fully_connected(fc, label.get_shape().as_list()[-1],
                                                   biases_initializer=tf.zeros_initializer(),
                                                   trainable=True, scope='full', reuse=reuse)
        end_points['Logits'] = logits

    if Distill is not None:
        if Distill == 'DML':
            teacher_train = True
            weight_decay = 5e-4
        else:
            is_training = False
            teacher_train = False
            weight_decay = 0.
        with tf.variable_scope('Teacher'):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                                variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'Teacher']):
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                                    param_regularizers={
                                                        'gamma': tf.contrib.layers.l2_regularizer(weight_decay),
                                                        'beta': tf.contrib.layers.l2_regularizer(weight_decay)},
                                                    variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'Teacher']):
                    n = 5
                    tch = tf.contrib.layers.conv2d(image, nChannels[0], [3, 3], 1, scope='conv0',
                                                   trainable=teacher_train, reuse=reuse)
                    tch = tf.contrib.layers.batch_norm(tch, scope='bn0', trainable=teacher_train,
                                                       is_training=is_training, reuse=reuse)
                    for i in range(len(stride)):
                        tch = NetworkBlock(tch, n, nChannels[1 + i], stride[i], is_training=is_training, reuse=reuse,
                                           name='N_%d' % i, stdScope=False)
                    fc = tf.reduce_mean(tch, [1, 2])
                    logits_tch = tf.contrib.layers.fully_connected(fc, label.get_shape().as_list()[-1],
                                                                   biases_initializer=tf.zeros_initializer(),
                                                                   trainable=teacher_train, scope='full', reuse=reuse)
                    end_points['Logits_tch'] = logits_tch

        with tf.variable_scope('Distillation'):
            feats = tf.get_collection('feat')
            #featT = tf.get_collection('feaT')
            print(len(feats))
            #print(len(featT))

            for i in range(len(feats)):
                print(feats[i])

            student_feats = feats[2]
            teacher_feats = feats[16]
            feats_noact = tf.get_collection('feat_noact')


            if Distill == 'FitNet':
                tf.add_to_collection('dist', Multiple.FitNet(student_feats, teacher_feats))

    return end_points
