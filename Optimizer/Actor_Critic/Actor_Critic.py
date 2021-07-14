from Optimizer.Actor_Critic.Actor_Critic_method import *
from tensorflow.keras.optimizers import Adagrad, Adam
from Optimizer.utils import CHECK_NOT_NONE


class Actor_Critic:
    def __init__(self, id=0, init_actor=init_actor, init_critic=init_critic,
                 nb_action=81, nb_state_features=10, name=None,
                 actor_learning_rate=1e-4, critic_learning_rate=1e-4,
                 gamma=0.9, network=None,
                 nb_status_features=5, alpha=0.1):

        self.id = id
        self.numActions = nb_action + 1
        self.numStateFeatures = nb_state_features
        self.ALR = actor_learning_rate
        self.CLR = critic_learning_rate
        self.gamma = gamma
        self.alpha = alpha
        self.name = name

        self.temp = 1
        self.step = 0
        self.lastState = None
        self.lastAction = None

        self.status_features = nb_status_features

        self.actor = init_actor(nb_state_features=nb_state_features, nb_actions=nb_action)
        self.critic = init_critic(nb_state_features=nb_state_features)

        self.actor_optimizer = Adagrad(learning_rate=self.ALR)
        self.critic_optimizer = Adam(learning_rate=self.CLR)

    def forward_actor(self, state):  # get policy
        CHECK_NOT_NONE(state, "forward_Actor", "Actor_Critic.py")
        return self.actor.predict(x=state, verbose=0)

    def forward_critic(self, state):  # get value
        CHECK_NOT_NONE(state, "forward_Critic", "Actor_Critic.py")
        return self.critic.predict(x=state, verbose=0)

    def update_local_optimizer(self, observation):
        CHECK_NOT_NONE_LIST(observation, "update_LocalOptimizer", "Actor_Critic.py")

        last_state, last_action, R, current_state = observation
        last_state_Value = self.forward_Critic(last_state)
        current_state_Value = 0
        if terminal_state(self, last_action):
            self.step = 0
            self.temp = 1
        else:
            current_state_Value = self.forward_Critic(current_state)

        temporal_diff = R + self.gamma * current_state_Value - last_state_Value
        train_ACnet(models=(self.actor, self.critic),
                    optimizers=(self.actor_optimizer, self.critic_optimizer),
                    input_state=last_state,
                    temporal_diff=temporal_diff,
                    gamma_factorial=self.temp,
                    steps=1)
        self.temp *= self.gamma

    def update(self, network, t):
        # extract current State
        current_state = extract_state_tensor(network)

        # If this is not the first step, then update local network
        if self.step != 0:
            last_state = self.lastState
            R = reward_function(network, self.lastAction)
            last_action = self.lastAction
            observation = (last_state, last_action, R, current_state)
            self.update_local_optimizer(observation)

        # If mc energy is below the threshold, then force recharging
        if network.mc_list[self.id].energy < 10:
            self.step += 1
            self.lastAction = self.numActions - 1
            self.lastState = current_state
            recharging_time = (network.mc_list[self.id].capacity - network.mc_list[self.id].energy) / network.mc_list[
                self.id].e_self_charge
            return self.lastAction, recharging_time

        # increase step
        self.step += 1
        self.lastAction = self.forward_critic(state=current_state)
        self.lastState = current_state

        # compute charging time
        charging_time = charging_time_func(mc_id=self.id, network=network,
                                           charging_pos_id=self.lastAction,
                                           time_stem=t, alpha=self.alpha)

        return self.lastAction, charging_time

    def __str__(self):
        self.actor.summary()
        print("Actor optimizer: " + str(self.actor_optimizer))
        print("Learning rate: %f" % self.ALR)
        self.critic.summary()
        print("Actor optimizer: " + str(self.critic_optimizer))
        print("Learning rate: %f" % self.CLR)

# if __name__ == "__main__":
#     ac_opt = Actor_Critic()
#     W = ac_opt.actor
#     net_weights_increase(W, 2)
#     net_weights_scale(ac_opt.actor, 3)
#     print(ac_opt.actor.trainable_weights[-1])
#     ac_opt.actor.trainable_weights[0] = ac_opt.actor.trainable_weights[0] + 1
#     print(W.trainable_weights[-1])
