from tensorflow.keras.layers import Dense, Dropout, Input, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from scipy.spatial import distance

from Optimizer.utils import *
import Simulator.parameter as para


def init_actor(nb_state_features=10, nb_actions=81, plot_figure=False):
    nodes_Input = nb_state_features
    nodes_Output = nb_actions
    # define network
    inputLayer = Input(shape=(nodes_Input,), name="State_Input")
    dense1 = Dense(units=256, activation='relu')(inputLayer)
    dropout1 = Dropout(rate=0.2)(dense1)
    dense2 = Dense(units=128, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.1)(dense2)
    dense3 = Dense(units=nodes_Output, activation='relu')(dropout2)
    output = Softmax(name="Policy")(dense3)
    # conclude network
    Actor_net = Model(inputs=[inputLayer], outputs=[output], name="Actor_network")
    if plot_figure:
        plot_model(Actor_net, show_shapes=True, show_layer_names=True, to_file="../models_figures/Actor_net.png")
    return Actor_net


def init_critic(nb_state_features, plot_figure=False):
    nodes_Input = nb_state_features
    # define network
    inputLayer = Input(shape=(nodes_Input,), name="State_Input")
    dense1 = Dense(units=256, activation='relu')(inputLayer)
    dropout1 = Dropout(rate=0.2)(dense1)
    dense2 = Dense(units=128, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.1)(dense2)
    dense3 = Dense(units=64, activation='relu')(dropout2)
    dropout3 = Dropout(rate=0.1)(dense3)
    dense4 = Dense(units=32, activation='relu')(dropout3)
    dropout4 = Dropout(rate=0.1)(dense4)
    dense5 = Dense(units=16, activation='relu')(dropout4)
    dropout5 = Dropout(rate=0.1)(dense5)
    dense6 = Dense(units=1, activation='relu')(dropout5)
    output = Softmax(name="Policy")(dense6)
    # conclude network
    Critic_net = Model(inputs=[inputLayer], outputs=[output], name="Critic_network")
    if plot_figure:
        plot_model(Critic_net, show_shapes=True, show_layer_names=True, to_file="../models_figures/Critic_net.png")
    return Critic_net


def actor_loss_function(y_pred, gamma_factorial, temporal_diff):
    return lambda: - np.log(y_pred) * gamma_factorial * temporal_diff


def critic_loss_function(y_pred, temporal_diff):
    return lambda: - y_pred * temporal_diff


# train network
"""
:parameter models = (actor, critic)
           input_state: current state
           optimizer = (actor_optimizer, critic_optimizer)
           temporal_diff = reward  + gamma * V(S') - V(S)
"""


@tf.function
def train_ACnet(models, input_state, optimizers, temporal_diff, gamma_factorial, steps=1):
    CHECK_NOT_NONE_LIST([models], "train_ACnet", "Actor_Critic_method.py")
    CHECK_NOT_NONE_LIST([optimizers], "train_ACnet", "Actor_Critic_method.py")
    CHECK_NOT_NONE(input_state, "train_ACnet", "Actor_Critic_method.py")

    actor, critic = models
    actor_opt, critic_opt = optimizers

    policy_loss = actor_loss_function(actor(input_state), gamma_factorial, temporal_diff)  # calculate lost
    value_loss = critic_loss_function(critic(input_state), temporal_diff)

    for step in range(steps):
        actor_opt.minimize(policy_loss, actor.trainable_weights)  # one step minimization
        critic_opt.minimize(value_loss, critic.trainable_weights)


# TODO: define the reward
def reward_function(network):

    pass


# TODO: define the terminal state
def terminal_state(ac, last_action):
    if last_action == ac.numAction - 1:
        return True
    return False


# TODO: get state from network
def extract_state_tensor(network):
    state = None
    # Implement here

    # return Tensor form of the state
    return tf.Tensor(state, dtype=tf.float32)


# TODO: get charging time
def charging_time_func(mc_id=0, network=None, charging_pos_id=None, time_stem=0, alpha=0.1):
    charging_position = network.charging_pos[charging_pos_id]
    mc = network.mc_list[mc_id]
    time_move = distance.euclidean(mc.current, charging_position) / mc.velocity
    energy_min = network.node[0].energy_thresh + alpha * network.node[0].energy_max
    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    for node in network.node:
        d = distance.euclidean(charging_position, node.location)
        p = para.alpha / (d + para.beta) ** 2
        p1 = 0
        for other_mc in network.mc_list:
            if other_mc.id != mc.id and other_mc.get_status() == "charging":
                d = distance.euclidean(other_mc.current, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2) * (other_mc.end_time - time_stem)
        if node.energy - time_move * node.avg_energy + p1 < energy_min and p - node.avg_energy > 0:
            s1.append((node.id, p, p1))
        if node.energy - time_move * node.avg_energy + p1 > energy_min and p - node.avg_energy < 0:
            s2.append((node.id, p, p1))
    t = []

    for index, p, p1 in s1:
        t.append((energy_min - network.node[index].energy + time_move * network.node[index].avg_energy - p1) / (
                p - network.node[index].avg_energy))
    for index, p, p1 in s2:
        t.append((energy_min - network.node[index].energy + time_move * network.node[index].avg_energy - p1) / (
                p - network.node[index].avg_energy))
    dead_list = []
    for item in t:
        nb_dead = 0
        for index, p, p1 in s1:
            temp = network.node[index].energy - time_move * network.node[index].avg_energy + p1 + (
                    p - network.node[index].avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        for index, p, p1 in s2:
            temp = network.node[index].energy - time_move * network.node[index].avg_energy + p1 + (
                    p - network.node[index].avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        dead_list.append(nb_dead)
    if dead_list:
        arg_min = np.argmin(dead_list)
        return t[arg_min]
    return 0

# if __name__ == "__main__":
# actor_net = init_Actor(nb_state_features=128,nb_actions=81)
# critic_net = init_Critic(nb_state_features=128)
# print(np.log(np.e))
