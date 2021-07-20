import torch
import numpy as np
import Simulator.parameter as para
from Optimizer.A3C.Server import Server
from Optimizer.A3C.Worker_method import reward_function, TERMINAL_STATE, \
    extract_state_tensor, charging_time_func, asynchronize


class Worker(Server):  # Optimizer
    def __init__(self, Server_object, name, id):
        super(Worker, self).__init__(nb_state_feature=Server_object.nb_state_feature,
                                     nb_action=Server_object.nb_action,
                                     name=name)

        self.id = id
        self.step = 0

        self.beta_entropy = para.A3C_beta_entropy
        self.gamma = para.A3C_gamma
        # self.k_step = para.A3C_k_step
        self.state_record = []  # record states
        self.reward_record = []  # record rewards

        self.action_space = [i for i in range(self.nb_action)]

    def get_policy(self, state_vector):
        return self.actor_net(state_vector)

    def get_value(self, state_vector):
        return self.critic_net(state_vector)

    def policy_loss_fn(self, policy, temporal_diff):
        return - temporal_diff[0] * torch.sum(torch.log(policy)) #- self.entropy_loss_fn(policy)

    def entropy_loss_fn(self, policy):
        return - self.beta_entropy * torch.sum(policy * torch.log(policy))

    def value_loss_fn(self, value, reward):
        return 1 / 2 * torch.pow(reward - value, 2)

    def accumulate_gradient(self):
        assert len(self.state_record) == len(self.reward_record) + 1, \
            "INVALID calling accumulate_gradient"

        R = 0 if TERMINAL_STATE(self.state_record[-1]) \
            else self.critic_net(self.state_record[-1])

        t = len(self.reward_record)
        for i in range(t):
            j = (t - 1) - i
            R = self.reward_record[j] + self.gamma * R

            state_vector = self.state_record[j]
            value = self.critic_net(state_vector)
            policy = self.actor_net(state_vector)

            value_loss = self.value_loss_fn(value=value, reward=R)
            value_loss.backward(retain_graph=True)

            tmp_diff = (R - value)
            policy_loss = self.policy_loss_fn(policy=policy, temporal_diff=tmp_diff.detach().numpy())
            policy_loss.backward(retain_graph=True)

    def reset_grad(self):
        self.actor_net.zero_grad()
        self.critic_net.zero_grad()

    def get_action(self, network=None, mc=None, time_stem=None):
        # state_record = [S(t), S(t+1), S(t+2)]
        # reward_record = [     R(t+1), R(t+2)]
        if len(self.state_record) != 0:
            R = reward_function(network)
            self.reward_record.append(R)

        state_tensor = extract_state_tensor(self, network)
        self.state_record.append(state_tensor)

        policy = self.get_policy(state_tensor)
        if torch.sum(policy) > 1:
            FILE = open("log/debug.txt", "w")
            FILE.write(np.array2string(state_tensor.detach().numpy()))
            FILE.write("\n")
            FILE.write(np.array2string(policy.detach().numpy()))
            FILE.close()

        action = np.random.choice(self.action_space, p=policy.detach().numpy())
        print(f"Here at location ({mc.current[0]}, {mc.current[1]}) worker id_{self.id} made decision")
        if action == self.nb_action - 1:
            return action, (mc.capacity - mc.energy) / mc.e_self_charge
        return action, charging_time_func(mc=mc, network=network, charging_pos_id=action, time_stem=time_stem)


if __name__ == "__main__":
    # action_space = [1, 2, 3, 4, 5]
    # a = torch.Tensor(action_space)
    # print(a.detach().numpy())

    b = torch.Tensor([3])
    b.requires_grad = True
    a = torch.Tensor([4])
    a.requires_grad = True
    c = 3*b + a
    c.backward(retain_graph=True)
    c.backward(retain_graph=True)
    print(b.grad)
    print(a.grad)
    print(b.detach().numpy()[0])
