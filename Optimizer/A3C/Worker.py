import torch
import numpy as np
import Simulator.parameter as para
from Optimizer.A3C.Server import Server
from Optimizer.A3C.Worker_method import reward_function, TERMINAL_STATE, \
    extract_state_tensor, charging_time_func, get_heuristic_policy, \
    extract_state_tensor_v2, one_hot

import csv


class Worker(Server):  # Optimizer
    def __init__(self, Server_object, name, id):
        super(Worker, self).__init__(nb_state_feature=Server_object.nb_state_feature,
                                     nb_action=Server_object.nb_action,
                                     name=name)

        self.id = id
        self.step = 0

        self.beta_entropy = para.A3C_beta_entropy
        self.gamma = para.A3C_gamma
        self.alpha_H = para.A3C_alpha_heuristic
        self.theta_H = para.A3C_decay_heuristic

        self.buffer = []
        self.action_space = [i for i in range(self.nb_action)]

    def create_experience(self, state, action,
                          policy_prob, behavior_prob,
                          reward = None):
        experience = {
            "step": self.step,              # record step t
            "reward": reward,               # record reward observe in this state
            "state": state,                 # record state vector t
            "action": action,               # record action at step t
            "policy_prob": policy_prob,     # record policy prob(action t)
            "behavior_prob": behavior_prob  # record behavior prob(action t)
        }

        with open(f"log/Worker_{self.id}.csv", mode="a+") as dumpfile:
            dumpfile_writer = csv.writer(dumpfile)
            if self.step == 0:
                dumpfile_writer.writerow(experience.keys())
            dumpfile_writer.writerow([
                self.step, reward, state.detach().numpy(), action, policy_prob.detach().numpy(), behavior_prob
            ])

        self.step += 1
        return experience

    def get_policy(self, state_vector):
        return self.actor_net(state_vector)

    def get_value(self, state_vector):
        return self.critic_net(state_vector)

    def policy_loss_fn(self, policy, action, temporal_diff):
        return - torch.sum(temporal_diff[0] * torch.log(policy) * torch.Tensor(one_hot(size=self.nb_action,
                                                                index=action))) \
               - self.entropy_loss_fn(policy)

    def entropy_loss_fn(self, policy):
        return - self.beta_entropy * torch.sum(policy * torch.log(policy))

    def value_loss_fn(self, value, reward):
        return (1/2) * torch.pow(reward - value, 2)

    def accumulate_gradient(self):
        R = 0 if TERMINAL_STATE(self.buffer[-1]["state"]) \
            else self.critic_net(self.buffer[-1]["state"])

        t = len(self.buffer) - 1        # i.e. (R,S,A) = [(S0,A0),(R1,S1,A1),(R2,S2,A2)]
        for i in range(t):              # 0, 1
            j = t - i             # 1, 0
            R = self.buffer[j]["reward"] + self.gamma * R

            state_vector = self.buffer[j]["state"]
            value = self.critic_net(state_vector)
            policy = self.actor_net(state_vector)

            value_loss = self.value_loss_fn(value=value, reward=R)
            value_loss.backward(retain_graph=True)

            tmp_diff = R - value
            policy_loss = self.policy_loss_fn(policy=policy,
                                              temporal_diff=tmp_diff.detach().numpy(),
                                              action=self.buffer[j]["action"])
            policy_loss.backward(retain_graph=True)

    def reset_grad(self):
        self.actor_net.zero_grad()
        self.critic_net.zero_grad()

    def get_action(self, network=None, mc=None, time_stem=None):
        R = None
        if self.step != 0:
            R = reward_function(network)

        state_tensor = extract_state_tensor(self, network)
        policy = self.get_policy(state_tensor)
        if torch.isnan(policy).any():
            FILE = open("debug.txt", "w")
            FILE.write(np.array2string(state_tensor.detach().numpy()))
            FILE.write("\n")
            FILE.write(np.array2string(policy.detach().numpy()))
            FILE.close()
            print("Error Nan policy")
            exit(100)

        # heuristic_policy = get_heuristic_policy(network=network, mc=mc, Worker=self)
        # assert np.sum(heuristic_policy) == 1, "Heuristic policy is false (sum not equals to 1)"

        # behavior_policy = (1 - self.alpha_H) * policy + self.alpha_H * heuristic_policy
        action = np.random.choice(self.action_space, p=policy.detach().numpy())

        # record all transitioning and reward
        self.buffer.append(
            self.create_experience(
                state=state_tensor, action=action,
                policy_prob=policy[action], behavior_prob= None, # behavior_policy[action],
                reward=R
            )
        )

        print(f"Here at location ({mc.current[0]}, {mc.current[1]}) worker id_{self.id} made decision")
        if action == self.nb_action - 1:
            return action, (mc.capacity - mc.energy) / mc.e_self_charge
        return action, charging_time_func(mc=mc, network=network, charging_pos_id=action, time_stem=time_stem)


if __name__ == "__main__":
    action_space = torch.Tensor([1, 2, 3, 4])
    print(action_space[1])
    # a = torch.Tensor(action_space)
    # print(a.detach().numpy())

    # b = torch.Tensor([3])
    # b.requires_grad = True
    # a = torch.Tensor([4])
    # a.requires_grad = True
    # c = 3*b + a
    # c.backward(retain_graph=True)
    # c.backward(retain_graph=True)
    # print(b.grad)
    # print(a.grad)
    # print(b.detach().numpy()[0])
    # alpha_H = 0.8
    # a = torch.Tensor([0.1, 0.2, 0.3, 0.4])
    # b = np.array([4,3,1,2])
    # c = np.exp(b)/np.sum(np.exp(b))
    # d = (1 - alpha_H) * a + alpha_H * c
    # print(d)
    # print(np.random.choice(action_space, p=d.detach().numpy()))







