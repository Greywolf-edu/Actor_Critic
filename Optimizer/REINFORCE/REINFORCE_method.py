from Optimizer.utils import *
import tensorflow as tf


def init_episodic_record():
    return {"States": [], "Actions": [], "Rewards": [None]}


def get_one_record(episode, index):
    return episode["States"][index], episode["Actions"][index], episode["Rewards"][index]


def actor_LossFunction(y_pred, gamma_factorial, temporal_diff):
    return lambda: - np.log(y_pred) * gamma_factorial * temporal_diff


def critic_LossFunction(y_pred, temporal_diff):
    return lambda: - y_pred * temporal_diff


@tf.function
def train_REINFORCE_net(models, input_state, optimizers, temporal_diff, gamma_factorial, steps=1):
    CHECK_NOT_NONE_LIST([models], "train_REINFORCE_net", "REINFORCE_method.py")
    CHECK_NOT_NONE_LIST([optimizers], "train_REINFORCE_net", "REINFORCE_method.py")
    CHECK_NOT_NONE(input_state, "train_REINFORCE_net", "REINFORCE_method.py")

    policy, value = models
    policy_opt, value_opt = optimizers

    policy_lost = actor_LossFunction(y_pred=policy(input_state),
                                     gamma_factorial=gamma_factorial,
                                     temporal_diff=temporal_diff)  # calculate lost

    loss_value = critic_LossFunction(y_pred=value(input_state),
                                     temporal_diff=temporal_diff)

    for step in range(steps):
        policy_opt.minimize(policy_lost, policy.trainable_weights)  # one step minimization
        value_opt.minimize(loss_value, value.trainable_weights)


# TODO: get reward from network
def reward_function(network):
    pass


# TODO: get state from network
def extract_state_tensor(network):
    state = None
    # Implement here

    # return Tensor form of the state
    return tf.Tensor(state, dtype=tf.float32)


# TODO: define the terminal state
def TERMINAL_STATE(state):
    pass


# TODO: get charging time
def charging_time_func(Object=None, network=None):
    pass