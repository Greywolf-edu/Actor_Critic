import torch
import numpy as np
from Optimizer.A3C.Server import Server
from Optimizer.A3C.Worker_method import reward_function, TERMINAL_STATE, \
    extract_state_tensor, charging_time_func, asynchronize


class Worker(Server):
    def __init__(self, Server_object, name, id,
                 actor_lr=1e-4, critic_lr=1e-4,
                 gamma = 0.95, beta_entropy=1e-2):
        super(Worker, self).__init__(nb_state_feature=Server_object.nb_state_feature,
                                     nb_action=Server_object.nb_action,
                                     name= name)

        self.id = id

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.beta_entropy = beta_entropy
        self.step = 0
        self.gamma = gamma
        self.k_step = 3
        self.state_record = []  # record k_step states
        self.reward_record = [] # record k_step rewards

        self.action_space = [i for i in range(self.nb_action)]

    def get_policy(self, state_vector):
        return self.actor_net(state_vector)

    def get_value(self, state_vector):
        return self.critic_net(state_vector)

    def policy_loss_fn(self, policy, temporal_diff):
        return - temporal_diff * torch.sum(torch.log(policy)) - self.entropy_loss_fn(policy)

    def entropy_loss_fn(self, policy):
        return - self.beta_entropy * torch.sum(policy * torch.log(policy))

    def value_loss_fn(self, value, reward):
        return 1/2 * torch.pow(reward - value,2)

    def accumulate_gradient(self, Server):
        R = 0 if TERMINAL_STATE(self.state_record[-1]) \
            else self.critic_net(self.state_record[-1])

        t = len(self.reward_record)
        for i in range(t):
            j = (t - 1) - i
            R = self.reward_record[j] + self.gamma * R

            state_vector = self.state_record[j]
            value = self.critic_net(state_vector)
            policy = self.actor_net(state_vector)

            value_loss = self.value_loss_fn(value=value,reward=R)
            value_loss.backward()

            tmp_diff = np.array(R - value)
            policy_loss = self.policy_loss_fn(policy=policy, temporal_diff=tmp_diff)
            policy_loss.backward()

        asynchronize(Server=Server, Worker=self)      # perform asynchronize
        self.reset_grad()   # clean gradient

    def reset_grad(self):
        self.actor_net.zero_grad()
        self.critic_net.zero_grad()

    def update(self, network):
        # state_record = [S(t), S(t+1), S(t+2)]
        # reward_record = [R(t+1), R(t+2), R(t+3)]
        state_tensor = extract_state_tensor(network)
        self.state_record.append(state_tensor)
        if self.step != 0:
            R = reward_function(network)
            self.reward_record.append(R)

        # Done k step or reach terminal state
        if len(self.state_record) == self.k_step or TERMINAL_STATE(state_tensor):
            assert len(self.state_record) == len(self.reward_record) + 1, \
                "INVALID calling accumulate_gradient"
            self.accumulate_gradient(network.global_optimizer) # Server
            # clear record
            self.reward_record.clear()
            self.state_record.clear()

        policy = self.get_policy(state_tensor)
        action = np.random.choice(self.action_space, p=policy)

        # increase step
        self.step += 1
        return action, charging_time_func(network)




if __name__ == "__main__":
    a = torch.tensor(6.0, requires_grad=True)
    b = torch.tensor(5.0, requires_grad=True)
    c = a + 3*b
    c.backward()
    print(c)
    print(a.grad)
    print(b.grad)
