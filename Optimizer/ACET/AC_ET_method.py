from Optimizer.utils import *


def actor_LossFunction(y_pred):
    return tf.math.log(y_pred)


def critic_LossFunction(y_pred):
    return y_pred


"""
:parameter models = (actor, critic)
           input_state: current state
           optimizer = (actor_optimizer, critic_optimizer)
           temporal_diff = reward  + gamma * V(S') - V(S)
"""
@tf.function
def train_ACET(models, optimizers, input_state, trace_vectorLists,
               trace_decays, gamma_factorial, temporal_diff,
               gamma=0.9, learning_rates=(1e-4, 1e-4)):

    actor, critic = models                      # models
    actor_opt, critic_opt = optimizers          # optimizers
    S = input_state                             # vector
    z_actor, z_critic = trace_vectorLists       # vectors
    lambda_actor, lambda_critic = trace_decays  # scalars
    actor_learning_rate, critic_learning_rate = learning_rates  # scalars

    # apply decay on trace vectorList:
    net_weights_scale(z_actor, gamma * lambda_actor)
    net_weights_scale(z_actor, gamma * lambda_critic)

    # get gradients of models
    with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
        policy = actor(S, training=True)
        value = critic(S, training=True)
        policy_loss = actor_LossFunction(policy)
        value_loss = critic_LossFunction(value)

    policy_grads = tape.gradient(policy_loss, actor.trainable_weights)
    value_grads = tape.gradient(value_loss, critic.trainable_weights)

    # perform increase trace vectorList:
    net_weights_scale(policy_grads, gamma_factorial)  # d(actor)/d(w) *= gamma_factorial
    net_weights_increase_v2(z_actor, policy_grads)    # z_ac = z_ac + d(actor)/d(w)
    net_weights_increase_v2(z_critic, value_grads)    # z_cr = z_cr + d(critic)/d(w)

    # make copy of trace vectorList:
    z_actor_copy = z_actor.copy()
    z_critic_copy = z_critic.copy()

    # z_temp = z * temporal
    net_weights_scale(z_actor_copy, temporal_diff)
    net_weights_scale(z_critic_copy, temporal_diff)

    # z_temp = z_temp * learning_rate
    net_weights_scale(z_actor_copy, actor_learning_rate)
    net_weights_scale(z_critic_copy, critic_learning_rate)

    # perform gradient accent on actor, critic weights
    net_weights_increase_v2(actor.trainable_weights, z_actor_copy)
    net_weights_increase_v2(critic.trainable_weights, z_critic_copy)

    del z_actor_copy
    del z_critic_copy
    del tape


# TODO: define the reward
def reward_function(network):
    pass


# TODO: define the terminal state
def TERMINAL_STATE(state):
    pass


# TODO: get state from network
def extract_state_tensor(network):
    state = None
    # Implement here

    # return Tensor form of the state
    return tf.Tensor(state, dtype=tf.float32)


# TODO: get charging time
def charging_time_func(Object=None, network=None):
    pass
