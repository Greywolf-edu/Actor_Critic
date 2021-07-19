import torch
from scipy.spatial import distance
import Simulator.parameter as para
import numpy as np
from Optimizer.A3C.Server_method import update_gradient


def get_nearest_charging_pos(current_location, charging_pos_list):
    MCpos = torch.Tensor(current_location)
    A = torch.Tensor(charging_pos_list)
    distance = torch.sqrt(torch.sum(torch.pow(MCpos - A, 2), dim=1))
    min_index = torch.argmin(distance)
    return charging_pos_list[min_index]


# TODO: define the reward
def reward_function(network):
    e_list = []
    for each_node in network.node:
        e_list.append(each_node.energy)
    return min(e_list)


# TODO: define the terminal state
def TERMINAL_STATE(state_tensor):
    # return state_tensor[0] == para.depot[0] and state_tensor[1] == para.depot[1]
    return False

# TODO: get state from network
def extract_state_tensor(worker, network):
    # Implement here
    MC = network.mc_list[worker.id]
    MC_location = [MC.current[0], MC.current[1]]    # get x, y coordination
    MC_infor = MC_location + [MC.energy]
    MC_infor_tensor = torch.Tensor(MC_infor)
    MC_infor_tensor = torch.flatten(MC_infor_tensor)  # flatten (3) [x, y, E]
    MC_infor_tensor.requires_grad = False

    charge_pos_infor = []
    charge_pos = network.charging_pos
    for mc in network.mc_list:
        if mc.id != MC.id:
            x_mc, y_mc = get_nearest_charging_pos(mc.current, charge_pos)
            e_mc = mc.energy
            charge_pos_infor.append([x_mc, y_mc, e_mc])
    charge_pos_tensor = torch.Tensor(charge_pos_infor)
    charge_pos_tensor = torch.flatten(charge_pos_tensor) # flatten (3 x nb_mc - 3)
    charge_pos_tensor.requires_grad = False

    nodes_infor = []
    for each_node in network.node:
        x, y = each_node.location
        E = each_node.energy # current energy
        e = each_node.avg_energy # consumsion rate
        nodes_infor.append([x, y, E, e])

    nodes_infor_tensor = torch.Tensor(nodes_infor)
    nodes_infor_tensor = torch.flatten(nodes_infor_tensor) # 4 x nb_node
    nodes_infor_tensor.requires_grad = False

    state = torch.cat([MC_infor_tensor, charge_pos_tensor, nodes_infor_tensor])
    # return Tensor form of the state
    return state # 3 x nb_mc + 4 x nb_node


def charging_time_func(mc_id=None, network=None, charging_pos_id=None, time_stem=0, alpha=0.1):
    """
    :param mc_id: index of current MC
    :param network
    :param charging_pos_id: index of Charging position
    :time_stem: current time stamp
    :alpha: hyper-parameter
    :return: duration time which the MC will stand charging for nodes
    """
    charging_position = network.charging_pos[charging_pos_id]
    mc = network.mc_list[mc_id]
    time_move = distance.euclidean(mc.current, charging_position) / mc.velocity
    energy_min = network.node[0].energy_thresh + alpha * network.node[0].energy_max
    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    for request in network.request_list:
        node = network.node[request["id"]] 
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


def asynchronize(Worker, Server): # MC sends gradient to Server
    """
    :param Worker: current MC's optimizer (self)
    :param Server: cloud
    This function perform asynchronize update to the cloud
    """
    Worker.accumulate_gradient()
    networks = (Worker.actor_net, Worker.critic_net)
    update_gradient(Server, networks)

    # clean gradient
    Worker.reset_grad()
    # clear record
    Worker.reward_record.clear()
    lastState= Worker.state_record[-1]
    Worker.state_record.clear()
    # restore current state for next use
    Worker.state_record.append(lastState)


def all_asynchronize(MCs, Server):
    """
    :param MCs: list of MC
    :param Server: cloud
    """
    for MC in MCs:
        asynchronize(MC.optimizer, Server)