def synchronize(Server, MCs):
    """
    This function synchronize all MC's networks with Server's
    :param Server: cloud
    :param MCs: list of MC
    """
    for MC in MCs:
        MC.optimizer.actor_net.load_state_dict(Server.actor_net.state_dict())
        MC.optimizer.critic_net.load_state_dict(Server.critic_net.state_dict())
        
        
def update_gradient(Server, MC_networks):
    """
    :param Server: cloud
    :param MC_networks: is ONE MC's networks, a tuple of (actor, critic)
    """
    MC_actor_net, MC_critic_net = MC_networks
    # Update server's actor network
    for serverParam, MCParam in \
            zip(Server.actor_net.parameters(), MC_actor_net.parameters()):
        serverParam.data += Server.actor_lr * MCParam.grad

    # Update server's critic network
    for serverParam, MCParam in \
            zip(Server.critic_net.parameters(), MC_critic_net.parameters()):
        serverParam.data += Server.critic_lr * MCParam.grad