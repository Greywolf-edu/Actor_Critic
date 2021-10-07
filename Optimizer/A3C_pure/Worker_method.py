import torch
import numpy as np
import Simulator.parameter as para
from scipy.spatial import distance


def get_nearest_charging_pos(current_location, charging_pos_list):
    mc_pos = torch.Tensor(current_location)
    A = torch.Tensor(charging_pos_list)
    distance_list = torch.sqrt(torch.sum(torch.pow(mc_pos - A, 2), dim=1))
    min_index = torch.argmin(distance_list)
    return charging_pos_list[min_index]

# TODO: re-define the reward
def reward_function(Worker, mc, network, time_stamp):
    r = -mc.last_distance_traveled / (1000 * np.sqrt(2)) + np.exp(mc.last_target_charged + 1) * mc.last_charging_energy_used

    record_file = open(f"log/{Worker.today}/{Worker.today_time_string}/worker@id{Worker.id}@reward-trace.csv", "a")
    record_file.write(f"{mc.last_distance_traveled}, {mc.last_target_charged}, {mc.last_charging_energy_used}, {r}\n")
    record_file.close()

    return r

# TODO: define terminal state
def TERMINAL_STATE(state_tensor):
    # return state_tensor[0] == para.depot[0] and state_tensor[1] == para.depot[1]
    return False

# TODO: define state
def extract_state_tensor(worker, net):
    def normalization(input, max, min):
        if np.isscalar(input):
            return [-1 + (input - min) * (1 - -1) / (max - min)]
        else:
            input = np.array(input)
            normalized_np = -1 + (input - min) * (1 - -1) / (max - min)
            return list(normalized_np)

    pos_max = 999
    pos_min = 0

    # Implement here
    MC = net.mc_list[worker.id]
    MC_location = [MC.current[0], MC.current[1]]  # get x, y coordination
    MC_location = normalization(MC_location, pos_max, pos_min)  # location normalization
    MC_energy = normalization(MC.energy, MC.capacity, 0)  # energy normalization
    MC_info = MC_location + MC_energy  # concat list

    MC_info_tensor = torch.Tensor(MC_info)
    MC_info_tensor = torch.flatten(MC_info_tensor)  # flatten (3) [x, y, E]
    MC_info_tensor.requires_grad = False

    charge_pos_info = []
    charge_pos = net.charging_pos
    for mc in net.mc_list:
        if mc.id != MC.id:
            x_mc, y_mc = get_nearest_charging_pos(mc.current, charge_pos)
            MC_location = normalization([x_mc, y_mc], pos_max, pos_min)  # location normalization
            e_mc = normalization(mc.energy, mc.capacity, 0)  # energy normalization
            charge_pos_info.append(MC_location + e_mc)  # concat list

    charge_pos_tensor = torch.Tensor(charge_pos_info)
    charge_pos_tensor = torch.flatten(charge_pos_tensor)  # flatten (3 x nb_mc - 3)
    charge_pos_tensor.requires_grad = False

    partition_info = []
    for i, pos in enumerate(charge_pos):
        charge_pos = normalization(pos, pos_max, pos_min)
        min_E = float('inf')
        max_e = float('-inf')

        for index_node in net.index_node_in_cluster[i]:
            node = net.node[index_node]
            if node.energy > 0:
                if node.energy < min_E:
                    min_E = node.energy
                if node.avg_energy > max_e:
                    max_e = node.avg_energy
        if min_E == float('inf'):
            min_E = 0
            max_e = 0
        min_E = normalization(min_E, 10, 0)
        max_e = normalization(max_e, 10, 0)
        partition_info.append(charge_pos + min_E + max_e)  # concat list

    partition_info_tensor = torch.Tensor(partition_info)
    partition_info_tensor = torch.flatten(partition_info_tensor)  # 4 x nb_action
    partition_info_tensor.requires_grad = False

    state = torch.cat([MC_info_tensor, charge_pos_tensor, partition_info_tensor])
    # return Tensor form of the state
    return state, [MC_info_tensor, charge_pos_tensor, partition_info_tensor]  # 3 x nb_mc, 4 x nb_charging_pos

# TODO: compute charging time
def charging_time_func(mc=None, net=None, action_id=None, time_stamp=0, theta=0.1):
    # return min(H_charging_time_func(mc=mc, net=net, action_id=action_id, time_stamp=time_stamp, theta=theta),
    #            para.A3C_max_charging_time)
    return 100


# heuristic timer
def H_charging_time_func(mc=None, net=None, action_id=None, time_stamp=0, theta=0.1):
    """
    :param mc: mobile charger
    :param net: network
    :param action_id: index of charging position
    :param time_stamp: current time stamp
    :param theta: hyper-parameter
    :return: duration time which the MC will stand charging for nodes
    """
    charging_position = net.charging_pos[action_id]
    time_move = distance.euclidean(mc.current, charging_position) / mc.velocity
    energy_min = net.node[0].energy_thresh + theta * net.node[0].energy_max
    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    # print(charging_position, len(net.request_id))
    for requesting_node in net.request_id:
        node = net.node[requesting_node]
        d = distance.euclidean(charging_position, node.location)
        p = para.alpha / (d + para.beta) ** 2
        p1 = 0
        for other_mc in net.mc_list:
            if other_mc.id != mc.id and other_mc.get_status != 'deactivated' and other_mc.end != para.depot:
                d = distance.euclidean(other_mc.current, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2) * \
                      (other_mc.end_time - max(time_stamp, other_mc.arriving_time))
        if node.energy - time_move * node.avg_energy + p1 < energy_min and p - node.avg_energy > 0:
            s1.append((node.id, p, p1))
        if node.energy - time_move * node.avg_energy + p1 > energy_min and p - node.avg_energy < 0:
            s2.append((node.id, p, p1))
    t = []

    for index, p, p1 in s1:
        t.append((energy_min - net.node[index].energy + time_move * net.node[index].avg_energy - p1) /
                 (p - net.node[index].avg_energy))
    for index, p, p1 in s2:
        t.append((energy_min - net.node[index].energy + time_move * net.node[index].avg_energy - p1) /
                 (p - net.node[index].avg_energy))
    dead_list = []
    for item in t:
        nb_dead = 0
        for index, p, p1 in s1:
            temp = net.node[index].energy - time_move * net.node[index].avg_energy + p1 + (
                    p - net.node[index].avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        for index, p, p1 in s2:
            temp = net.node[index].energy - time_move * net.node[index].avg_energy + p1 + (
                    p - net.node[index].avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        dead_list.append(nb_dead)
    if dead_list:
        arg_min = np.argmin(dead_list)
        return t[arg_min]
    else:
        return 0



