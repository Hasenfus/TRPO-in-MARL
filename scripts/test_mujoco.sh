#!/bin/sh
env="mujoco"
scenario="Ant-v3"
agent_conf="4x2"
agent_obsk=2
algo=$1
te=$2
de=$3
di=$4
exp="test"
model_dir=$5
kl_threshold=1e-4
num_trn_steps=$2
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
#comp comp
/home/pearl0/miniconda3/envs/TRPO/bin/python /home/pearl0/Desktop/MARL/TRPO-in-MARL/scripts/test/test_mujoco.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --lr 5e-6 --critic_lr 5e-3 --std_x_coef 1 --std_y_coef 2e-1 --running_id 1 --n_training_threads 8 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 100 --ppo_epoch 5 --kl_threshold ${kl_threshold} --use_value_active_masks --use_eval --add_center_xy --use_state_agent --share_policy --test_episodes ${te} --displayed_episodes ${de} --display_interval ${di} --model_dir ${model_dir}

#pers comp
#/Users/Hunter/miniconda3/envs/TRPO/bin/python /home/pearl0/Desktop/MARL/TRPO-in-MARL/scripts/test/test_mujoco.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --lr 5e-6 --critic_lr 5e-3 --std_x_coef 1 --std_y_coef 2e-1 --running_id ${number} --n_training_threads 8 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 100 --ppo_epoch 5 --kl_threshold ${kl_threshold} --use_value_active_masks --use_eval --add_center_xy --use_state_agent --share_policy --test_episodes ${te} --displayed_episodes ${de} --display_interval ${di}

