import torch.nn as nn
import Simulator.parameter as para
from Optimizer.A3C.Server_method import update_gradient, zero_net_weights


class Server(nn.Module):
    def __init__(self, nb_state_feature, nb_action, name):
        super(Server, self).__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(in_features=nb_state_feature, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=nb_action),
            nn.Softmax()
        )
        zero_net_weights(self.actor_net)

        self.critic_net = nn.Sequential(
            nn.Linear(in_features=nb_state_feature, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.ReLU()
        )
        zero_net_weights(self.critic_net)

        self.actor_lr = para.A3C_serverActor_lr
        self.critic_lr = para.A3C_serverCritic_lr

        self.nb_state_feature = nb_state_feature
        self.nb_action = nb_action

        self.name = name

    def update_gradient(self, MC_networks):
        update_gradient(self, MC_networks)


if __name__ == "__main__":
    a = (1, 2, 3, 4, 5)
    b = [4, 3, 5, 2, 6]
    c = zip(a, b)
    for x, y in c:
        print(x, y)
