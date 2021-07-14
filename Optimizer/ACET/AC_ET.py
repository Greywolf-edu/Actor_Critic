from Optimizer.ACET.AC_ET_method import *
from Optimizer.utils import *
from Optimizer.Actor_Critic.Actor_Critic import Actor_Critic


class AC_ET(Actor_Critic):
    def __init__(self, nb_actions, nb_state_features, nb_status_features, isEpisodic=True):
        super().__init__(nb_action=nb_actions, nb_state_features=nb_state_features,
                         nb_status_features=nb_status_features)

        self.isEpisodic = isEpisodic
        self.trace_decay_actor = 0.85
        self.trace_decay_critic = 0.85

        self.trace_actor_vectorList = self.critic.trainable_weights.copy()
        net_weights_scale(self.trace_actor_vectorList, 0)   # z_actor = vector 0
        self.trace_critic_vectorList = self.critic.trainable_weights.copy()
        net_weights_scale(self.trace_critic_vectorList, 0)  # z_critic = vector 0

    def update_LocalOptimizer(self, observation):
        CHECK_NOT_NONE_LIST(observation, "update_LocalOptimizer", "AC_ET.py")
        if self.isEpisodic:
            self.update_LocalEpisodic(observation)
        else:
            self.update_LocalContinuing(observation)

    def update(self, network):
        if self.isEpisodic:
            return self.episodic_update(network)
        else:
            return self.continuing_update(network)

    def episodic_update(self, network):
        # extract current State
        Scurr = extract_state_tensor(network)
        # If this is not the first step, then update local network
        if self.step != 0:
            Slast = self.lastState
            R = reward_function(network)
            Alast = self.lastAction
            observation = (Slast, Alast, R, Scurr)
            self.update_LocalOptimizer(observation)

        # compute charging time
        charging_time = charging_time_func(Object=self, network=network)
        # increase step then return
        self.step += 1
        return self.forward_Critic(state=Scurr), charging_time

    def continuing_update(self, network):
        pass

    def update_LocalEpisodic(self, observation):
        Slast, Alast, R, Scurr = observation
        last_state_Value = self.forward_Critic(Slast)
        current_state_Value = 0
        if TERMINAL_STATE(Scurr):
            self.step = 0
            self.temp = 1
        else:
            current_state_Value = self.forward_Critic(Scurr)

        temporal_diff = R + self.gamma * current_state_Value - last_state_Value

        train_ACET(models=(self.actor, self.critic),
                      optimizers=(self.actor_optimizer, self.critic_optimizer),
                      input_state=Slast,
                      temporal_diff=temporal_diff,
                      gamma_factorial=self.temp,
                      steps=1)
        self.temp *= self.gamma

    def update_LocalContinuing(self, observation):
        pass