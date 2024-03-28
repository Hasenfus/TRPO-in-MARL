#!/bin/sh
env="mujoco"
scenario="Ant-v2"
agent_conf="4x2"
agent_obsk=2
algo=$1
exp="mlp"
running_max=4
kl_threshold=1e-4
malfunction=$2
mal_agent=$3
mal_episode=$4
env_steps=$5
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in `seq ${running_max}`;
do
    echo "the ${number}-th running:"
    /home/pearl0/miniconda3/envs/TRPO/bin/python /home/pearl0/Desktop/MARL/TRPO-in-MARL/scripts/train/train_mujoco_malfunction.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --lr 5e-6 --critic_lr 5e-3 --std_x_coef 1 --std_y_coef 5e-1 --running_id ${number} --n_training_threads 8 --n_rollout_threads 4 --num_mini_batch 40 --episode_length 100 --num_env_steps ${env_steps} --ppo_epoch 5 --kl_threshold ${kl_threshold} --use_value_active_masks --use_eval --add_center_xy --use_state_agent --share_policy  --mal_agent ${mal_agent} --malfunction ${malfunction} --mal_episode ${mal_episode}
done
