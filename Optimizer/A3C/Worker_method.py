import torch
from scipy.spatial import distance
import Simulator.parameter as para
import numpy as np
from Optimizer.A3C.Server_method import update_gradient


def get_nearest_charging_pos(current_location, charging_pos_list):
    mc_pos = torch.Tensor(current_location)
    A = torch.Tensor(charging_pos_list)
    distance_list = torch.sqrt(torch.sum(torch.pow(mc_pos - A, 2), dim=1))
    min_index = torch.argmin(distance_list)
    return charging_pos_list[min_index]


# TODO: re-define the reward
def reward_function(network):
    e_list = []
    for each_node in network.node:
        if each_node.energy > 0:
            e_list.append(each_node.energy)

    print("Average energy of living nodes: " + str(np.mean(np.array(e_list))))
    return min(e_list)


def TERMINAL_STATE(state_tensor):
    # return state_tensor[0] == para.depot[0] and state_tensor[1] == para.depot[1]
    return False


# TODO: get state from network (modified - Hoang Hai Long)
def extract_state_tensor(worker, network):
    # Implement here
    MC = network.mc_list[worker.id]
    MC_location = [MC.current[0], MC.current[1]]  # get x, y coordination
    MC_info = MC_location + [MC.energy]
    MC_info_tensor = torch.Tensor(MC_info)
    MC_info_tensor = torch.flatten(MC_info_tensor)  # flatten (3) [x, y, E]
    MC_info_tensor.requires_grad = False

    charge_pos_info = []
    charge_pos = network.charging_pos
    for mc in network.mc_list:
        if mc.id != MC.id:
            x_mc, y_mc = get_nearest_charging_pos(mc.current, charge_pos)
            e_mc = mc.energy
            charge_pos_info.append([x_mc, y_mc, e_mc])
    charge_pos_tensor = torch.Tensor(charge_pos_info)
    charge_pos_tensor = torch.flatten(charge_pos_tensor)  # flatten (3 x nb_mc - 3)
    charge_pos_tensor.requires_grad = False

    # nodes_info = []
    # for each_node in network.node:
    #     x, y = each_node.location
    #     E = each_node.energy  # current energy
    #     e = each_node.avg_energy  # consumption rate
    #     nodes_info.append([x, y, E, e])
    #
    # nodes_info_tensor = torch.Tensor(nodes_info)
    # nodes_info_tensor = torch.flatten(nodes_info_tensor)  # 4 x nb_node
    # nodes_info_tensor.requires_grad = False

    partition_info = []
    r = 10
    for pos in charge_pos:
        x_charge_pos, y_charge_pos = pos
        min_E = float('inf')
        max_e = float('-inf')
        for each_node in network.node:
            if each_node.is_activate and torch.dist(torch.tensor(each_node.location, dtype=torch.float),
                                                    torch.tensor(pos, dtype=torch.float)) <= r:
                if each_node.energy < min_E:
                    min_E = each_node.energy
                if each_node.avg_energy > max_e:
                    max_e = each_node.avg_energy
        partition_info.append([x_charge_pos, y_charge_pos, min_E, max_e])

    partition_info_tensor = torch.Tensor(partition_info)
    partition_info_tensor = torch.flatten(partition_info_tensor)  # 3 x nb_action
    partition_info_tensor.requires_grad = False

    # state = torch.cat([MC_info_tensor, charge_pos_tensor, nodes_info_tensor])
    state = torch.cat([MC_info_tensor, charge_pos_tensor, partition_info])
    # return Tensor form of the state
    return state  # 3 x nb_mc + 4 x nb_node


# TODO: get state from network (new - Nguyen Thanh Long)
def extract_state_tensor_v2(worker, network):
    return None


def charging_time_func(mc=None, network=None, charging_pos_id=None, time_stem=0, alpha=0.1):
    """
    :param mc: mobile charger
    :param network: network
    :param charging_pos_id: index of charging position
    :param time_stem: current time stamp
    :param alpha: hyper-parameter
    :return: duration time which the MC will stand charging for nodes
    """
    charging_position = network.charging_pos[charging_pos_id]
    time_move = distance.euclidean(mc.current, charging_position) / mc.velocity
    energy_min = network.node[0].energy_thresh + alpha * network.node[0].energy_max
    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    # print(charging_position, len(network.request_id))
    for request_node_id in network.request_id:
        node = network.node[request_node_id]
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
    else:
        return 0


# TODO: impelement heuristic policy (Nguyen Thanh Long)
def get_heuristic_policy(mc=None, Worker=None, network=None):
    H_policy = None  # numpy array of size = #nb_action

    return H_policy


def one_hot(index, size):
    one_hot_vector = np.zeros(size)
    one_hot_vector[index] = 1
    one_hot_vector = torch.Tensor(one_hot_vector)
    one_hot_vector.requires_grad = False
    return one_hot_vector


def asynchronize(Worker, Server, time_step=None):  # MC sends gradient to Server
    """
    :param Worker: current MC's optimizer (self)
    :param Server: cloud
    This function perform asynchronize update to the cloud
    """
    print(f"Worker id_{Worker.id} asynchronized with len(buffer): {len(Worker.buffer)}")
    if len(Worker.buffer) >= 2:
        Worker.accumulate_gradient(timestep=time_step)
        networks = (Worker.actor_net, Worker.critic_net)
        update_gradient(Server, networks)

        # clean gradient
        Worker.reset_grad()
        # clear record
        lastBuffer = Worker.buffer[-1]
        Worker.buffer.clear()
        # restore current state for next use
        Worker.buffer.append(lastBuffer)
    else:
        print(f"Worker id_{Worker.id} has nothing to asynchronize")


def all_asynchronize(MCs, Server, moment=None):
    """
    :param moment: current time step
    :param MCs: list of MC
    :param Server: cloud
    """
    print("All asynchronize!")
    for MC in MCs:
        asynchronize(Worker=MC.optimizer, Server=Server, time_step=moment)
