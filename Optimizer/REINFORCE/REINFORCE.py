from Optimizer.Actor_Critic import Actor_Critic
from Optimizer.Actor_Critic.Actor_Critic_method import init_Actor, init_Critic
from Optimizer.REINFORCE.REINFORCE_method import *


class REINFORCE(Actor_Critic.Actor_Critic):
    def __init__(self):
        super(REINFORCE, self).__init__(init_actor=init_Actor, init_critic=init_Critic, nb_action=81,
                                        nb_state_features=10, actor_learning_rate=1e-4, critic_learning_rate=1e-4,
                                        gamma=0.9)
        self.episode = init_episodic_record()
        self.step = 0

    def update(self, network):
        Scurr = extract_state_tensor(network)
        if TERMINAL_STATE(Scurr):
            self.episode["States"].append(None) # S(T)
            self.episode["Rewards"].append(0)   # R(T)

            # execute update network
            self.update_LocalOptimizer()
            # reset agent to a new episode
            self.episode = init_episodic_record()
            self.step = 0
        else:
            self.episode["States"].append(Scurr)

        if self.step != 0:
            reward = reward_function(network)
            self.episode["Rewards"].append(reward)

        # make action
        action = self.actor.predict(x=Scurr)
        self.episode["Actions"].append(action)

        charging_time = charging_time_func(Object=self, network=network)
        self.step += 1
        return action, charging_time

    def update_LocalOptimizer(self, observation=None):
        # episode: (R0 = None), S0, A0, R1, S1, A1, R2, S2, A2, ...., R(T-1), S(T-1), A(T-1) = "go charge", R(T), S(T) = Terminal (None)
        T = len(self.episode["States"]) - 1  # Not updating terminal state
        J = 1
        for t in range(T):  # t = 0, 1, 2, ... T-1
            G = 0
            I = 1
            for k in range(t + 1, T + 1):  # k = t+1, t+2, t+3, ... ,T-1, T
                G += I * self.episode["Rewards"][k]
                I *= self.gamma

            temporal_diff = G - self.forward_Critic(self.episode["State"][t])
            train_REINFORCE_net(models=(self.actor, self.critic),
                                input_state=self.episode["State"][t],
                                optimizers=(self.actor_optimizer, self.critic_optimizer),
                                temporal_diff=temporal_diff,
                                gamma_factorial=J,
                                steps=2)
            J *= self.gamma


if __name__ == "__main__":
    pass