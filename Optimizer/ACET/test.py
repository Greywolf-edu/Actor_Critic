from Optimizer.Actor_Critic.Actor_Critic import Actor_Critic
from Optimizer.utils import *
from Optimizer.ACET.AC_ET_method import actor_LossFunction, critic_LossFunction


ac_agent = Actor_Critic(nb_action=4, nb_state_features=2)
print(ac_agent.actor.trainable_weights[-1])

S = tf.constant([[1,2]])

with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
    policy = ac_agent.actor(S, training=True)
    value = ac_agent.critic(S, training=True)
    policy_loss = actor_LossFunction(policy)
    value_loss = critic_LossFunction(value)

policy_grads = tape.gradient(policy_loss, ac_agent.actor.trainable_weights)
value_grads = tape.gradient(value_loss, ac_agent.critic.trainable_weights)

print("gradient printing....")
print(policy_grads[-1])

print("Updating ...")
net_weights_scale(policy_grads, 1e-2)
net_weights_increase_v2(ac_agent.actor.trainable_weights, policy_grads)
print(ac_agent.actor.trainable_weights[-1])

del tape
#
# print(policy_grads)