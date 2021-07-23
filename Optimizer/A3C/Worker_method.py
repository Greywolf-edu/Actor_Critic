import torch
from scipy.spatial import distance
import Simulator.parameter as para
import numpy as np
from Optimizer.A3C.Server_method import update_gradient
from Simulator.Sensor_Node.node_method import find_receiver


def get_nearest_charging_pos(current_location, charging_pos_list):
    mc_pos = torch.Tensor(current_location)
    A = torch.Tensor(charging_pos_list)
    distance_list = torch.sqrt(torch.sum(torch.pow(mc_pos - A, 2), dim=1))
    min_index = torch.argmin(distance_list)
    return charging_pos_list[min_index]


# TODO: re-define the reward
def reward_function(net):
    e_list = []
    for each_node in net.node:
        if each_node.energy > 0:
            e_list.append(each_node.energy)

    print("Average energy of living nodes: " + str(np.mean(np.array(e_list))))
    return min(e_list)


def TERMINAL_STATE(state_tensor):
    # return state_tensor[0] == para.depot[0] and state_tensor[1] == para.depot[1]
    return False


def extract_state_tensor(worker, net):
    # Implement here
    MC = net.mc_list[worker.id]
    MC_location = [MC.current[0], MC.current[1]]  # get x, y coordination
    MC_info = MC_location + [MC.energy]
    MC_info_tensor = torch.Tensor(MC_info)
    MC_info_tensor = torch.flatten(MC_info_tensor)  # flatten (3) [x, y, E]
    MC_info_tensor.requires_grad = False

    charge_pos_info = []
    charge_pos = net.charging_pos
    for mc in net.mc_list:
        if mc.id != MC.id:
            x_mc, y_mc = get_nearest_charging_pos(mc.current, charge_pos)
            e_mc = mc.energy
            charge_pos_info.append([x_mc, y_mc, e_mc])
    charge_pos_tensor = torch.Tensor(charge_pos_info)
    charge_pos_tensor = torch.flatten(charge_pos_tensor)  # flatten (3 x nb_mc - 3)
    charge_pos_tensor.requires_grad = False

    partition_info = []
    for i, pos in enumerate(charge_pos):
        x_charge_pos, y_charge_pos = pos
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

        partition_info.append([x_charge_pos, y_charge_pos, min_E, max_e])

    partition_info_tensor = torch.Tensor(partition_info)
    partition_info_tensor = torch.flatten(partition_info_tensor)  # 4 x nb_action
    partition_info_tensor.requires_grad = False

    state = torch.cat([MC_info_tensor, charge_pos_tensor, partition_info_tensor])
    # return Tensor form of the state
    return state  # 3 x nb_mc + 4 x nb_charging_pos


# TODO: get state from network (new - Nguyen Thanh Long)
def extract_state_tensor_v2(worker, net):
    return None


def charging_time_func(mc=None, net=None, action_id=None, time_stem=0, theta=0.1):
    """
    :param mc: mobile charger
    :param net: network
    :param action_id: index of charging position
    :param time_stem: current time stamp
    :param theta: hyper-parameter
    :return: duration time which the MC will stand charging for nodes
    """
    charging_position = net.charging_pos[action_id]
    time_move = distance.euclidean(mc.current, charging_position) / mc.velocity
    energy_min = net.node[0].energy_thresh + theta * net.node[0].energy_max
    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    # print(charging_position, len(net.request_id))
    for request_node_id in net.request_id:
        node = net.node[request_node_id]
        d = distance.euclidean(charging_position, node.location)
        p = para.alpha / (d + para.beta) ** 2
        p1 = 0
        for other_mc in net.mc_list:
            if other_mc.id != mc.id and other_mc.get_status() == "charging":
                d = distance.euclidean(other_mc.current, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2) * (other_mc.end_time - time_stem)
        if node.energy - time_move * node.avg_energy + p1 < energy_min and p - node.avg_energy > 0:
            s1.append((node.id, p, p1))
        if node.energy - time_move * node.avg_energy + p1 > energy_min and p - node.avg_energy < 0:
            s2.append((node.id, p, p1))
    t = []

    for index, p, p1 in s1:
        t.append((energy_min - net.node[index].energy + time_move * net.node[index].avg_energy - p1) / (
                p - net.node[index].avg_energy))
    for index, p, p1 in s2:
        t.append((energy_min - net.node[index].energy + time_move * net.node[index].avg_energy - p1) / (
                p - net.node[index].avg_energy))
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


# TODO: implement heuristic policy (Nguyen Thanh Long)
def get_heuristic_policy(net=None, mc=None, worker=None, time_stem=0):
    energy_factor = torch.ones_like(torch.Tensor(worker.action_space))
    priority_factor = torch.ones_like(torch.Tensor(worker.action_space))
    target_monitoring_factor = torch.ones_like(torch.Tensor(worker.action_space))
    self_charging_factor = torch.ones_like(torch.Tensor(worker.action_space))
    for action_id in worker.action_space:
        temp = heuristic_function(net=net, mc=mc, optimizer=worker, action_id=action_id, time_stem=time_stem)
        energy_factor[action_id] = temp[0]
        priority_factor[action_id] = temp[1]
        target_monitoring_factor[action_id] = temp[2]
        self_charging_factor[action_id] = temp[3]
    energy_factor = energy_factor/torch.sum(energy_factor)
    priority_factor = priority_factor/torch.sum(priority_factor)
    target_monitoring_factor = target_monitoring_factor/torch.sum(target_monitoring_factor)
    self_charging_factor = self_charging_factor/torch.sum(self_charging_factor)
    H_policy = energy_factor + priority_factor + target_monitoring_factor - self_charging_factor
    H_policy = torch.softmax(H_policy, 0)
    H_policy.requires_grad = False
    return H_policy  # torch tensor size = #nb_action


def heuristic_function(net=None, mc=None, optimizer=None, action_id=0, time_stem=0, receive_func=find_receiver):
    if action_id == optimizer.nb_action - 1:
        return 0, 0, 0, 0
    theta = optimizer.charging_time_theta
    charging_time = charging_time_func(mc, net, action_id=action_id, time_stem=time_stem,
                                       theta=theta)
    w, nb_target_alive = get_weight(net=net, mc=mc, action_id=action_id, charging_time=charging_time,
                                    receive_func=receive_func)
    p = get_charge_per_sec(net=net, action_id=action_id)
    p_hat = p / np.sum(p)
    E = np.asarray([net.node[request["id"]].energy for request in net.request_list])
    e = np.asarray([request["avg_energy"] for request in net.request_list])
    third = nb_target_alive / len(net.target)
    second = np.sum(w * p_hat)
    first = np.sum(e * p / E)
    forth = (mc.capacity - (mc.energy - charging_time*p))/mc.capacity
    return first, second, third, forth


def get_weight(net, mc, action_id, charging_time, receive_func=find_receiver):
    p = get_charge_per_sec(net, action_id)
    all_path = get_all_path(net, receive_func)
    time_move = distance.euclidean(mc.current,
                                   net.charging_pos[action_id]) / mc.velocity
    list_dead = []
    w = [0 for _ in net.list_request]
    for request_id, request in enumerate(net.request_id):
        temp = (net.node[request["id"]].energy - time_move * request["avg_energy"]) + (
                p[request_id] - request["avg_energy"]) * charging_time
        if temp < 0:
            list_dead.append(request["id"])
    for request_id, request in enumerate(net.request_id):
        nb_path = 0
        for path in all_path:
            if request["id"] in path:
                nb_path += 1
        w[request_id] = nb_path
    total_weight = sum(w) + len(w) * 10 ** -3
    w = np.asarray([(item + 10 ** -3) / total_weight for item in w])
    nb_target_alive = 0
    for path in all_path:
        if para.base in path and not (set(list_dead) & set(path)):
            nb_target_alive += 1
    return w, nb_target_alive


def get_charge_per_sec(net=None, action_id=None):
    return np.asarray(
        [para.alpha / (distance.euclidean(net.node[request["id"]].location,
                                          net.charging_pos[action_id]) + para.beta) ** 2 for
         request in net.list_request])


def get_path(net, sensor_id, receive_func=find_receiver):
    path = [sensor_id]
    if distance.euclidean(net.node[sensor_id].location, para.base) <= net.node[sensor_id].com_ran:
        path.append(para.base)
    else:
        receive_id = receive_func(net=net, node=net.node[sensor_id])
        if receive_id != -1:
            path.extend(get_path(net, receive_id, receive_func))
    return path


def get_all_path(net, receive_func=find_receiver):
    list_path = []
    for sensor_id, target_id in enumerate(net.target):
        list_path.append(get_path(net, sensor_id, receive_func))
    return list_path


def one_hot(index, size):
    one_hot_vector = np.zeros(size)
    one_hot_vector[index] = 1
    one_hot_vector = torch.Tensor(one_hot_vector)
    one_hot_vector.requires_grad = False
    return one_hot_vector


def tensor2value(tensor):
    return tensor.detach().numpy()


def asynchronize(Worker, Server, time_step=None):  # MC sends gradient to Server
    """
    :param time_step:
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
