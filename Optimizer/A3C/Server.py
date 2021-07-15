import torch.nn as nn
from Optimizer.A3C.Server_method import update_gradient


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

        self.actor_lr = 1e-4
        self.critic_lr = 1e-3

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
