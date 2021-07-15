import torch
from Simulator.parameter import depot


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
    MC_infor = [MC.current[0], MC.current[1], MC.energy]
    MC_infor_tensor = torch.Tensor(MC_infor)
    MC_infor_tensor.requires_grad = False

    charging_pos_infor = []
    for charging_pos in network.nb_charging_pos:
        charging_pos_infor.append(None) #????
    charging_pos_tensor = torch.Tensor(charging_pos_infor)
    charging_pos_tensor.requires_grad = False

    nodes_infor = []
    for each_node in network.node:
        x, y = each_node.location
        E = each_node.energy
        e = each_node.avg_energy
        nodes_infor.append([x, y, E, e])

    nodes_infor_tensor = torch.Tensor(nodes_infor) # shape = (numnode, 4)
    nodes_infor_tensor.requires_grad = False

    state = None
    # return Tensor form of the state
    return torch.Tensor(state, dtype=torch.float32)


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
    Server.update_gradient(networks)
