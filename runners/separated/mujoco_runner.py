import time
import numpy as np
from functools import reduce
import torch
from runners.separated.base_runner import Runner
import os
import pickle
import time

def _t2n(x):
    return x.detach().cpu().numpy()


class MujocoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""


    def __init__(self, config):
        super(MujocoRunner, self).__init__(config)

    def run(self, malfunction = False, mal_episode = 30_000, malagent_old = None, malagent_new = None):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        mal_episode = int(mal_episode) // self.episode_length // self.n_rollout_threads
        print(f"mal_episode: {mal_episode}")

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        break_leg = False
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            if episode > mal_episode and malfunction:
                    break_leg = True
                    # print("malfunctioning agent is ", malagent_old)
                    # print("malfunctioning agent is ", malagent_new)

            done_episodes_rewards = []
            # print(episode)
            for step in range(self.episode_length):
                # print(step)
                # Sample actions
                if malfunction and break_leg:
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step, malagent_old, malagent_new)
                else:
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, _ = self.envs.step(actions)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t] or step == self.episode_length - 1:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                data = obs, share_obs, rewards, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("some episodes done, average rewards: ", aver_episode_rewards)
                    self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards},
                                             total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                print("Evaluating")
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step, malagent_old = None, malagent_new = None):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        old_agent_action = None
        new_agent_action = None
        if malagent_old == -1 and malagent_new == 1:
            malagent_old = 0
        if malagent_old == -1 and malagent_new == 3:
            malagent_old = 2
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            value_collector.append(_t2n(value))
            #

            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        # [self.envs, agents, dim]

        if malagent_old is not None:
            if malagent_new is not None:
                action_collector[malagent_old] = action_collector[malagent_new]
                action_collector[malagent_new] = np.zeros_like(action_collector[malagent_new])
                action_log_prob_collector[malagent_old] = action_log_prob_collector[malagent_new]
                action_log_prob_collector[malagent_new] = np.zeros_like(action_log_prob_collector[malagent_new])
            else:
                action_collector[malagent_old] = np.zeros_like(action_collector[malagent_old])
                action_log_prob_collector[malagent_old] = np.zeros_like(action_log_prob_collector[malagent_old])


        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], None,
                                         active_masks[:, agent_id], None)

    def log_train(self, train_infos, total_num_steps):
        print("average_step_rewards is {}.".format(np.mean(self.buffer[0].rewards)))
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)



    @torch.no_grad()
    def test(self, test_episodes=10):
        self.warmup()

        all_trajectories = []
        all_last_x = []
        all_last_xy = []
        all_tot_dist = []
        final_ep_rewards = []

        reward_forward_all = []
        reward_forward = 0.0
        reward_ctrl_all = []
        reward_ctrl = 0.0
        reward_survive_all = []
        reward_survive = 0.0
        healthy_rewards = 0.0
        healthy_rewards_all = []
        reward_contact = 0.0
        reward_contact_all = []

        print("Running tests")
        for episode in range(test_episodes):
            done_episodes_rewards = []
            trajectory = []

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                obs, share_obs, rewards, dones, infos, _ = self.envs.step(actions)

                dones_env = np.all(dones, axis=1)
                # print(dones_env.shape)
                reward_env = np.mean(rewards, axis=1).flatten()
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(reward_env[t])

                info_dict = infos[0][0]
                # print(infos, rewards)
                reward_forward += info_dict['reward_forward']
                reward_ctrl += info_dict['reward_ctrl']
                reward_survive += info_dict['reward_survive']
                reward_contact += info_dict['reward_contact']
                # print(type(infos), infos, info_dict)
                # print(obs)
                # print(share_obs)
                trajectory.append((info_dict['x_position'], info_dict['y_position']))
                healthy_rewards += rewards

                if np.any(dones_env):
                    break
            print(f"Episode {episode} finished after {reward_forward - reward_ctrl + reward_survive}")
            healthy_rewards_all.append(healthy_rewards)
            reward_forward_all.append(reward_forward)
            reward_ctrl_all.append(reward_ctrl)
            reward_survive_all.append(reward_survive)
            reward_contact_all.append(reward_contact)
            reward_forward = 0.0
            reward_ctrl = 0.0
            reward_survive = 0.0
            healthy_rewards = 0.0
            reward_contact = 0.0
            final_ep_rewards.append(np.mean(done_episodes_rewards))
            all_trajectories.append(trajectory)
            # all_last_x.append(info_dict['x_position'])
            # all_last_xy.append((info_dict['x_position'], info_dict['y_position']))
            all_tot_dist.append(info_dict['distance_from_origin'])

        directory_name_with_time = time.strftime("%Y%m%d-%H%M%S")
        full_directory_path = os.path.join(self.test_dir, directory_name_with_time)

        if not os.path.exists(full_directory_path):
            os.makedirs(full_directory_path)

        rew_file_name = os.path.join(full_directory_path, 'test' + '_healthy_rewards.pkl')
        with open(rew_file_name, 'wb') as fp:
            pickle.dump(final_ep_rewards, fp)

        trajectories_file_name = os.path.join(full_directory_path,
                                              'test' + '_healthy_trajectories.pkl')
        with open(trajectories_file_name, 'wb') as fp:
            pickle.dump(all_trajectories, fp)

        distances_file_name = os.path.join(full_directory_path,
                                           'test' + '_healthy_distances.pkl')
        with open(distances_file_name, 'wb') as fp:
            pickle.dump(all_tot_dist, fp)

        reward_for_file_name = os.path.join(full_directory_path,
                                           'test' + '_reward_forward.pkl')
        with open(reward_for_file_name, 'wb') as fp:
            pickle.dump(reward_forward_all, fp)

        reward_survive_file_name = os.path.join(full_directory_path,
                                           'test' + '_reward_survive.pkl')
        with open(reward_survive_file_name, 'wb') as fp:
            pickle.dump(reward_survive_all, fp)

        reward_ctrl_file_name = os.path.join(full_directory_path,
                                           'test' + '_reward_ctrl.pkl')
        with open(reward_ctrl_file_name, 'wb') as fp:
            pickle.dump(reward_ctrl_all, fp)

        reward_contact_file_name = os.path.join(full_directory_path,
                                        'test' + '_reward_contact.pkl')
        with open(reward_contact_file_name, 'wb') as fp:
            pickle.dump(reward_contact_all, fp)

        reward_file_name = os.path.join(full_directory_path,
                                             'test' + '_healthy_rewards.pkl')
        with open(reward_file_name, 'wb') as fp:
            pickle.dump(reward_ctrl_all, fp)
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        episode_t = 0
        while True:
            # print(eval_episode)
            eval_actions_collector = []
            eval_rnn_states_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                      eval_rnn_states[:, agent_id],
                                                      eval_masks[:, agent_id],
                                                      deterministic=True)
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions)
            episode_t += 1
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i] or episode_t < self.episode_length:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards,
                                  'eval_max_episode_rewards': [np.max(eval_episode_rewards)]}
                self.log_env(eval_env_infos, total_num_steps)
                print("eval_average_episode_rewards is {}.".format(np.mean(eval_episode_rewards)))
                break
