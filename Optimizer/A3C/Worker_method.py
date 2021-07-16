import torch
from Simulator.parameter import depot
from Server_method import update_gradient


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
    return state_tensor[0] == depot[0] and state_tensor[1] == depot[1]


# TODO: get state from network
def extract_state_tensor(worker, network):
    # Implement here
    MC = network.mc_list[worker.id]
    MC_location = [MC.current[0], MC.current[1]]
    MC_infor = MC_location + [MC.energy]
    MC_infor_tensor = torch.Tensor(MC_infor)
    MC_infor_tensor = torch.flatten(MC_infor_tensor)  # flatten
    MC_infor_tensor.requires_grad = False

    charge_pos_infor = []
    charge_pos = network.charging_pos
    for mc in network.mc_list:
        if mc.id != MC.id:
            x_mc, y_mc = get_nearest_charging_pos(mc.current, charge_pos)
            e_mc = mc.energy
            charge_pos_infor.append([x_mc, y_mc, e_mc])
    charge_pos_tensor = torch.Tensor(charge_pos_infor)
    charge_pos_tensor = torch.flatten(charge_pos_tensor) # flatten
    charge_pos_tensor.requires_grad = False

    nodes_infor = []
    for each_node in network.node:
        x, y = each_node.location
        E = each_node.energy
        e = each_node.avg_energy
        nodes_infor.append([x, y, E, e])

    nodes_infor_tensor = torch.Tensor(nodes_infor)  # shape
    nodes_infor_tensor = torch.flatten(nodes_infor_tensor)
    nodes_infor_tensor.requires_grad = False

    state = torch.cat([MC_infor_tensor, charge_pos_tensor, nodes_infor_tensor])
    # return Tensor form of the state
    return state


# TODO: get charging time
def charging_time_func(Object=None, network=None):
    pass


def asynchronize(Worker, Server):
    """
    :param Worker: current MC (self)
    :param Server: cloud
    This function perform asynchronize update to the cloud
    """
    networks = (Worker.actor_net, Worker.critic_net)
    update_gradient(Server, networks)
