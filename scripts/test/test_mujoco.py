#!/usr/bin/env python
import sys
import os
sys.path.append("../")
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from configs.config import get_config
from envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from runners.separated.mujoco_runner import MujocoRunner as Runner
"""Train script for Mujoco."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mujoco":
                env_args = {"scenario": all_args.scenario,
                            "env_version": 3,
                            "agent_conf": all_args.agent_conf,
                            "agent_obsk": all_args.agent_obsk,
                            "episode_limit": all_args.episode_length,
                            "healthy_reward":0.1,
                            "terminate_when_unhealthy":False,
                            "use_contact_forces":False}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mujoco":
                env_args = {"scenario": all_args.scenario,
                            "env_version": 3,
                            "agent_conf": all_args.agent_conf,
                            "agent_obsk": all_args.agent_obsk,
                            "episode_limit": all_args.episode_length,
                            "healthy_reward":0.1,
                            "terminate_when_unhealthy":False,
                            "use_contact_forces":False}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='Hopper-v2', help="Which mujoco-old task to run on")
    parser.add_argument('--agent_conf', type=str, default='3x1')
    parser.add_argument('--agent_obsk', type=int, default=0)
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)

    # agent-specific state should be designed carefully
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)
    parser.add_argument("--use_single_network", action='store_true', default=False)
    parser.add_argument("--episode_len", type=int, default=100)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--test_episodes", type=int, default=100)
    parser.add_argument("--displayed_episodes", type=int, default=2)
    parser.add_argument("--display_interval", type=int, default=50)
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed=all_args.running_id
    else:
        all_args.seed=np.random.randint(1000,10000)
    print("seed is :",all_args.seed)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        print(f"device count :: {torch.cuda.device_count()}")
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
        print(f"device count :: {torch.cuda.device_count()}")

    run_dir = Path(all_args.model_dir) / "test"
    all_args.model_dir = all_args.model_dir + "/models"

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = 4
    print("num_agents:", num_agents)

    config = {
        "all_args": all_args,
        "envs": envs,
        "experiment_name": all_args.experiment_name,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "test_dir": run_dir / "test"
    }

    # run experiments
    runner = Runner(config)
    runner.test(test_episodes = all_args.test_episodes)

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    # runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
