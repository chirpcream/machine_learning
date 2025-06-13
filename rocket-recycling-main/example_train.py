import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as plt
import utils
import os
import glob
import cv2
import csv #tyq

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'hover'  # 'hover' or 'landing'
    rocket_type = 'falcon'  #SunYunru:考虑变量rocket_type:可以选'falcon'或'starship'
    version = '_raw'  #SunYunru:增设变量version，方便对比不同修改下代码运行结果
    record_video = True  #SunYunru:增设变量record_video，确定是否保存视频

    max_m_episode = 20000  #SunYunru:改到20000轮训练
    max_steps = 800

    #env = Rocket(task=task, max_steps=max_steps, rocket_type=rocket_type)
    # tyq
    env = Rocket(task=task, max_steps=max_steps, rocket_type=rocket_type,
             wind_enabled=True, wind_force_max=2.0,
             mass_init=2, fuel_mass=1.8)

    ckpt_folder = os.path.join('./', task + '_' + rocket_type + version + '_ckpt_222')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

        # 记录奖励情况 tyq
    log_path = os.path.join(ckpt_folder, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'dist', 'pose', 'fuel_bonus', 'landing_bonus', 'crash_penalty', 'fuel_left', 'step', 'landed', 'crashed'])

    last_episode_id = 0
    REWARDS = []

    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1], weights_only=False)  #SunYunru:兼容不同版本的pytorch
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']

    for episode_id in range(last_episode_id, max_m_episode):
        if episode_id % 1000 == 1 and record_video:  #SunYunru:设置视频保存功能，每1000轮训练保存一次视频
            video_path = os.path.join(ckpt_folder, f'train_ep_{episode_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 2/env.dt, (env.viewport_w, env.viewport_h))
        # training loop
        state = env.reset()
        rewards, log_probs, values, masks = [], [], [], []
        for step_id in range(max_steps):
            action, log_prob, value = net.get_action(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)
            if episode_id % 1000 == 1 and record_video:
                frame_0, frame_1 = env.render()
                video_writer.write(cv2.cvtColor(frame_0, cv2.COLOR_RGB2BGR))
                video_writer.write(cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR))
            elif episode_id % 500 == 1:
                env.render()
            if done or step_id == max_steps-1:
                _, _, Qval = net.get_action(state)
                net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=0.999)
                break
        if episode_id % 1000 == 1 and record_video and 'video_writer' in locals():
            video_writer.release()

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))
        
        # tyq 记录奖励
        reward_parts = env._last_reward_parts if hasattr(env, '_last_reward_parts') else {}

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode_id,
                reward_parts.get('total_reward', 0),
                reward_parts.get('dist_reward', 0),
                reward_parts.get('pose_reward', 0),
                reward_parts.get('fuel_bonus', 0),
                reward_parts.get('landing_bonus', 0),
                reward_parts.get('crash_penalty', 0),
                reward_parts.get('fuel_left', 0),
                reward_parts.get('step_id', 0),
                reward_parts.get('landed', False),
                reward_parts.get('crashed', False)
            ])


        if episode_id % 100 == 1:
            plt.figure()
            plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'), dpi=1200)  #SunYunru:提高输出图像的清晰度
            plt.close()

            torch.save({'episode_id': episode_id,
                        'REWARDS': REWARDS,
                        'model_G_state_dict': net.state_dict()},
                       os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))



