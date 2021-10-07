import Simulator.parameter as para
import numpy as np
import torch
from Optimizer.A3C_pure.Worker_method import reward_function, TERMINAL_STATE, \
    extract_state_tensor, charging_time_func

from datetime import date, datetime
import os


class Worker:
    def __init__(self, Server=None, id=None, theta=None, today=None, time=None):
        self.id = id
        self.buffer = []
        self.action_space = [i for i in range(Server.nb_action)]
        self.cummulate_loss = 0
        self.heuristic_theta = theta

        self.server = Server
        self.step = 0

        record_file = open(f"log/debug_worker/worker@id{self.id}.csv", "w")
        record_file.write(f"step, reward, action, charging_time, proba\n")
        record_file.close()

        self.today = today
        self.today_time_string = time

        os.system(f"mkdir log/{self.today}/{self.today_time_string}")

        record_file = open(f"log/{self.today}/{self.today_time_string}/worker@id{self.id}@reward-trace.csv", "w")
        record_file.write("distance_traveled, nb_target_charged, energy_charged, reward\n")
        record_file.close()

    def create_experience(self, state=None, action=None, charging_time=None,
                          policy_prob=None, behavior_prob=None,
                          reward=None):
        experience = {
            "step": self.step,              # record step t
            "reward": reward,               # record reward observe in this state
            "state": state,                 # record state vector t
            "action": action,               # record action at step t
            "charging_time": charging_time, # record charging time of the action
            "policy_prob": policy_prob,     # record policy prob(action t)
            "behavior_prob": behavior_prob  # record behavior prob(action t)
        }

        record_file = open(f"log/debug_worker/worker@id{self.id}.csv", "a")
        record_file.write(f"{self.step},{reward},{action},{charging_time},{policy_prob}\n")
        record_file.close()

        self.step += 1
        return experience
    

    def get_action(self, network=None, mc=None, time_stamp=None):
        R = 0
        if len(self.buffer):
            R = reward_function(Worker=self, mc=mc, network=network, time_stamp=time_stamp) \
                if self.buffer[-1]["charging_time"] > 0 else para.A3C_bad_reward

        state_tensor, dont_care = extract_state_tensor(self, network)
        policy = self.server.get_policy(state_tensor)

        action = np.random.choice(self.action_space, p=policy.detach().numpy())

        # get charging time
        if action == self.server.nb_action - 1:
            charging_time = (mc.capacity - mc.energy) / mc.e_self_charge
        else:
            charging_time = charging_time_func(mc=mc, net=network, action_id=action, time_stamp=time_stamp,
                                          theta=self.heuristic_theta)

        # record all transitioning and reward
        self.buffer.append(
            self.create_experience(
                state=state_tensor,
                action=action,
                charging_time=charging_time,
                policy_prob=policy[action].detach().numpy(),
                reward=R
            )
        )

        # reset metrics for mobile charger
        mc.last_target_charged = 0
        mc.last_charging_energy_used = 0
        mc.last_distance_traveled = 0

        return action, charging_time


    def policy_loss_fn(self, policy, action, temporal_diff):
        return temporal_diff * torch.log(policy[action])


    def entropy_loss_fn(self, policy):
        return para.A3C_beta_entropy * torch.mean(policy * torch.log(policy))


    def value_loss_fn(self, value, reward):
        return - (1/2) * torch.pow(reward - value, 2)


    def accumulate_loss(self):
        # i.e. (R,S,A) = [(R0,S0,A0),(R1,S1,A1),(R2,S2,A2)]
        t = len(self.buffer) - 1        # t = 2
        if t <= 0:
            return False

        R = 0 if TERMINAL_STATE(self.buffer[-1]["state"]) \
            else self.server.get_value(self.buffer[-1]["state"])     # R[2]

        for i in range(t):              # 0, 1
            j = t - 1 - i               # 1, 0
            R = self.buffer[j+1]["reward"] + para.A3C_gamma * R  # r = R[2] + gamma * V(S[2])

            state_vector = self.buffer[j]["state"]
            value = self.server.get_value(state_vector)
            policy = self.server.get_policy(state_vector)

            entropy_loss = self.entropy_loss_fn(policy=policy)
            value_loss = self.value_loss_fn(value=value, reward=R)
            total_value_loss = value_loss - entropy_loss

            value = value.detach() if torch.is_tensor(value) else value

            tmp_diff = R - value

            policy_loss = self.policy_loss_fn(policy=policy,
                                              temporal_diff=tmp_diff,
                                              action=self.buffer[j]["action"])

            self.cummulate_loss += total_value_loss + policy_loss
        
        return True

    
    def update_server(self):
        has_experience = self.accumulate_loss()
        self.server.update_gradient_server(self.cummulate_loss)
        if has_experience:
            self.buffer.clear()
        self.cummulate_loss = 0

