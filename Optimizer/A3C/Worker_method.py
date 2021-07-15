import torch


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
