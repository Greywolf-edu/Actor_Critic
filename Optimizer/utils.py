import numpy as np
import tensorflow as tf


def CHECK_NOT_NONE(val, func_name, filename):
    if val is None:
        print(str(val) + ": NONE VALUE PASSED IN FUNCTION " + func_name + " in file " + filename + "!")
        exit(120)


def CHECK_NOT_NONE_LIST(list, func_name, filename):
    CHECK_NOT_NONE(list, func_name, filename)
    for val in list:
        CHECK_NOT_NONE(val, func_name, filename)


def CHECK_SCALAR(val, func_name, filename):
    CHECK_NOT_NONE(val, func_name, filename)
    if not np.isscalar(val):
        print(str(val) + " NOT SCALAR VALUE PASSED IN FUNCTION: " + func_name + " in file " + filename + "!")
        exit(121)


def net_weights_scale(trainable_weights, scalar):
    for i in range(len(trainable_weights)):
        trainable_weights[i] = tf.multiply(trainable_weights[i], scalar)


def net_weights_increase(trainable_weights, scalar):
    for i in range(len(trainable_weights)):
        trainable_weights[i] = tf.add(trainable_weights[i], scalar[i])


def net_weights_increase_v2(trainable_weights, vector_list):
    assert len(trainable_weights) == len(vector_list), "INVALID in net_weights_increase_v2"
    for i in range(len(trainable_weights)):
        trainable_weights[i].assign(trainable_weights[i] + vector_list[i])