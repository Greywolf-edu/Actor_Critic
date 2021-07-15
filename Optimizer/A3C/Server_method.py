import copy


def synchronize(Server, MCs):
    """
    This function synchronize all MC's networks with Server's
    :param Server: cloud
    :param MCs: list of MC
    :return: nothing
    """
    for MC in MCs:
        MC.actor_net.load_state_dict(Server.actor_net.state_dict())
        MC.critic_net.load_state_dict(Server.critic_net.state_dict())