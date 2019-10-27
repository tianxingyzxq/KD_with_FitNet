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




def NetworkBlock(x,nb_layers, depth,stride,is_training=False,reuse=False, name='',
                 guidedLayerIndex=3,nbrl=0,BottelneckChanelNBR=8,teacher=True,max_pool_h_w=[4,4]):

    for i in range(nb_layers):
        strd = stride if i == 0 else 1
        depth_modified=depth
        added_bottelneck_here = False
        if guidedLayerIndex == nbrl and teacher is False:
            added_bottelneck_here=True

        out=tf.contrib.layers.conv2d(x, depth_modified, [3, 3], stride=strd,scope=name + 'conv0%d' % i, trainable=True,
                                                                        reuse=reuse)
        nbrl = nbrl + 1
        tf.add_to_collection('feat', out)
        out = tf.contrib.layers.batch_norm(out, scope=name + 'bn0%d' % i, trainable=True, is_training=is_training,
                                           reuse=reuse)
        if guidedLayerIndex == nbrl and teacher is False:
            added_bottelneck_here = True

        out = tf.contrib.layers.conv2d(out, depth_modified, [3, 3], 1, scope=name + 'conv1%d' % i, trainable=True,
                                       reuse=reuse)
        nbrl = nbrl + 1
        tf.add_to_collection('feat', out)
        out = tf.contrib.layers.batch_norm(out, scope=name + 'bn1%d' % i, trainable=True, is_training=is_training,
                                           reuse=reuse,
                                           activation_fn=None)
        if strd > 1 or depth != x.get_shape().as_list()[-1]:
            x = tf.contrib.layers.conv2d(x, depth_modified, [1, 1], stride, scope=name + 'conv2%d' % i, trainable=True,
                                         reuse=reuse)
            x = tf.contrib.layers.batch_norm(x, scope=name + 'bn2%d' % i, trainable=True, is_training=is_training,
                                             reuse=reuse,
                                             activation_fn=None)
        out = x + out
        if added_bottelneck_here == True and teacher is False:
            out = tf.contrib.layers.conv2d(out, BottelneckChanelNBR, [3, 3], 1, scope=name + 'ConvBottelneck%d' % i, trainable=True,
                                           reuse=reuse)

            out = tf.contrib.layers.batch_norm(out, scope=name + 'BnormBottelneck%d' % i, trainable=True, is_training=is_training,
                                             reuse=reuse,
                                             activation_fn=None)
            out=tf.layers.max_pooling2d(
                out,
                max_pool_h_w,
                (2,2),
                padding='valid',
                data_format='channels_last',
                name='max_pool_bottelneck'
            )
            print('Bottelneck added')
        out = tf.nn.relu(out)
    return out,nbrl


def ResNet(image, label, scope, is_training, reuse=False, Distill=None,guidedLayerIndex=3,hintLayerIndex=11
           ,BottelneckChanelNBR=8,max_pool_h_w=[4,4]):
    end_points = {}
    nChannels = [32 , 64 , 128, 256]
    stride = [1, 2, 2]
    n = 1 if scope != 'Teacher' else 5
    teacher = True if scope == 'Teacher' else False
    with tf.variable_scope(scope):
        layerCompter = 0
        std = tf.contrib.layers.conv2d(image, nChannels[0], [3, 3], 1, scope='conv0', trainable=True, reuse=reuse)
        std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable=True, is_training=is_training, reuse=reuse)
        print(len(stride))
        for i in range(len(stride)):
            print(' i %d'%i)
            print(' i %d' % (1+i))
            std,layerCompter = NetworkBlock(std, n, nChannels[1 + i], stride[i], is_training=is_training, reuse=reuse,
                               name='N%d' % i,guidedLayerIndex=guidedLayerIndex,nbrl=layerCompter,
                                            BottelneckChanelNBR=BottelneckChanelNBR,teacher=teacher,
                                            max_pool_h_w=max_pool_h_w)
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
                    layerCompter = 0
                    tch = tf.contrib.layers.conv2d(image, nChannels[0], [3, 3], 1, scope='conv0',
                                                   trainable=teacher_train, reuse=reuse)
                    tch = tf.contrib.layers.batch_norm(tch, scope='bn0', trainable=teacher_train,
                                                       is_training=is_training, reuse=reuse)

                    for i in range(len(stride)):
                        tch,layerCompter = NetworkBlock(tch, n, nChannels[1 + i], stride[i],
                                                        is_training=is_training, reuse=reuse,
                                           name='N_%d' % i, guidedLayerIndex=guidedLayerIndex,
                                                        BottelneckChanelNBR=BottelneckChanelNBR,
                                                        nbrl=layerCompter,teacher=True,max_pool_h_w=max_pool_h_w)
                    fc = tf.reduce_mean(tch, [1, 2])
                    logits_tch = tf.contrib.layers.fully_connected(fc, label.get_shape().as_list()[-1],
                                                                   biases_initializer=tf.zeros_initializer(),
                                                                   trainable=teacher_train, scope='full', reuse=reuse)
                    end_points['Logits_tch'] = logits_tch

        with tf.variable_scope('Distillation'):
            feats = tf.get_collection('feat')
            print(len(feats))
            for i in range(len(feats)):
                if i== hintLayerIndex+5:
                    print('[hintLayer]')
                if i== (guidedLayerIndex) :
                    print('[guidedLayer]')
                print(feats[i])

            student_feats = feats[guidedLayerIndex]
            teacher_feats = feats[hintLayerIndex+5]
            if Distill == 'FitNet':
                tf.add_to_collection('dist', Multiple.FitNet(student_feats, teacher_feats))
    return end_points
