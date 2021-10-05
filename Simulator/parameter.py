alpha = 36.0
beta = 30.0
base = (500.0, 500.0)
depot = (0.0, 0.0)
b = 500.0
b_energy = 0.0
ER = 0.0000001
ET = 0.00000005
EFS = 0.00000000001
EMP = 0.0000000000000013
prob = 1.0
E_mc_thresh = 10

# heuristic
charging_time_theta = 5

# A3C param
A3C_synchronize_T = 200
A3C_gamma = 0.95
A3C_beta_entropy = 0.01
A3C_clip_grad = 0.5

A3C_start_Body_lr = 2 * 1e-2
A3C_start_Actor_lr = 2 * 1e-2
A3C_start_Critic_lr = 2 * 1e-2
A3C_decay_lr = 0.9397455978
A3C_bad_reward = -1
A3C_max_charging_time = 500

# Simulation
SIM_duration = 5000
SIM_partition_time = 300
SIM_log_frequency = 100
SIM_plot_network = False

# Node
NODE_e_thresh_ratio = 0.6


# dumpfile
FILE_debug_update = "log/weight_record/param.txt"
FILE_debug_loss = "log/weight_record/loss.csv"
FILE_debug_Nan_heuristic = "log/weight_record/Nan_heurisitc.txt"


# model savings
MODEL_save = True
MODEL_load = False
MODEL_save_actor_path = "Model_weights/A3C-pure/actor"
MODEL_save_critic_path = "Model_weights/A3C-pure/critic"
MODEL_save_body_path = "Model_weights/A3C-pure/body"


