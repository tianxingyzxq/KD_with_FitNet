import tensorflow as tf

def FitNet(student, teacher):
    '''
     Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta,  and  Yoshua  Bengio.
     Fitnets:   Hints  for  thin  deep  nets.
     arXiv preprint arXiv:1412.6550, 2014.
    '''
    def Guided(source, target):
        with tf.variable_scope('Guided'):
            Ds = source.get_shape().as_list()[-1]
            Dt = target.get_shape().as_list()[-1]
            print(Ds)
            print(Dt)
            if Ds != Dt:
                with tf.variable_scope('Map'):
                    target = tf.contrib.layers.fully_connected(target, Ds, biases_initializer = None, trainable=True, scope = 'fc')
            return tf.reduce_mean(tf.square(source-target))
    return tf.add_n([Guided(student, teacher)])