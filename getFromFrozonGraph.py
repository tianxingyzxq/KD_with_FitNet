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

def main():
    # Get the frozen graph
    #the teacher graph
    frozen_graphTeacher = freeze_graph(home_path+"/teacher",home_path+"/teacher/TeacherHint.ckpt.meta", "hintLayerNext")
    #the student grap
    froszen_graphStuden= freeze_graph(home_path+"/student",home_path+"/student/studentGuided.ckpt.meta", "studentGuided")
    # Set the frozen graph as a default graph
    froszen_graphStuden.as_default()
    # Get the output tensor from the pre-trained model
    pre_trained_model_result = froszen_graphStuden.get_tensor_by_name("studentGuided")


def freeze_graph(model_dir, meta_graph_path, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                           comma separated
    """
    if not tf.io.gfile.exists(model_dir):
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
        saver = tf.train.import_meta_graph(meta_graph_path, clear_devices=clear_devices)

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
    main()
