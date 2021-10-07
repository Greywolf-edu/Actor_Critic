import torch
import torch.nn as nn
import Simulator.parameter as para
# from Optimizer.A3C_pure.Server_method import update_gradient

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Server():
    def __init__(self, nb_state_feature, nb_action, name):
        super(Server, self).__init__()

        self.body_net = nn.Sequential(
            nn.Linear(in_features=nb_state_feature, out_features=256),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=512),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.Tanh(),
            nn.Dropout(p=0.5),
        )

        self.actor_net = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU6(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU6(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=nb_action),
            nn.Softmax(dim=0)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=1),
            nn.ReLU()
        )

        self.body_net.apply(init_weights)
        self.actor_net.apply(init_weights)
        self.critic_net.apply(init_weights)

        self.Body_optimizer = torch.optim.Adagrad(self.body_net.parameters(), lr=para.A3C_start_Body_lr, lr_decay=para.A3C_decay_lr)
        self.Actor_optimizer = torch.optim.Adagrad(self.actor_net.parameters(), lr=para.A3C_start_Actor_lr, lr_decay=para.A3C_decay_lr)
        self.Critic_optimizer = torch.optim.Adagrad(self.critic_net.parameters(), lr=para.A3C_start_Critic_lr, lr_decay=para.A3C_decay_lr)

        self.net = [self.body_net, self.actor_net, self.critic_net]

        self.nb_state_feature = nb_state_feature
        self.nb_action = nb_action

        self.name = name

    def get_policy(self, state_vector):
        body_out = self.body_net(state_vector)
        return self.actor_net(body_out)

    def get_value(self, state_vector):
        body_out = self.body_net(state_vector)
        return self.critic_net(body_out)

    def update_gradient_server(self, MC_loss):
        if torch.is_tensor(MC_loss):
            self.Body_optimizer.zero_grad()
            self.Actor_optimizer.zero_grad()
            self.Critic_optimizer.zero_grad()

            MC_loss.backward()

            torch.nn.utils.clip_grad_value_(self.Body_optimizer.parameters(),para.A3C_clip_grad)
            torch.nn.utils.clip_grad_value_(self.Actor_optimizer.parameters(),para.A3C_clip_grad)
            torch.nn.utils.clip_grad_value_(self.Critic_optimizer.parameters(),para.A3C_clip_grad)

            self.Body_optimizer.step()
            self.Actor_optimizer.step()
            self.Critic_optimizer.step()

        

