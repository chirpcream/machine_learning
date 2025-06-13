import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as plt
import utils
import os
import glob
import cv2
import csv #TanYingqi:导入csv模块以便保存数据

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    #SunYunru:训练前必须注意的内容
    task = 'hover'  # 'hover' or 'landing'
    rocket_type = 'starship'  #SunYunru:考虑变量rocket_type:可以选'falcon'或'starship'
    version = '_wind&fuel'  #SunYunru:增设变量version，方便对比不同修改下代码运行结果:可以选'_raw'或'_wind&fuel'
    entropy_set =True  #TanYingqi:增设变量entropy_set，促进策略多样性探索  #SunYunru:整合完善
    layer_norm = True  #TanYingqi:增设变量layer_norm，确定是否使用层归一化  #SunYunru:整合完善
    record_video = True  #SunYunru:增设变量record_video，确定是否保存视频
    max_m_episode = 30000  #SunYunru:改到30000轮训练
    max_steps = 800

    #SunYunru:常规参数初始化
    ckpt_folder = os.path.join('./', task + '_' + rocket_type + version + '_ckpt')
    wind_enabled = False
    wind_force_max = 0.3
    fuel_mass = 5.0
    mass_init = 10.0
    fuel_consumption_rate = 0.01
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    last_episode_id = 0
    REWARDS = []

    #SunYunru:依据不同version进行参数修改
    if version == '_wind&fuel':
        wind_enabled = True
        wind_force_max = 0.3
        fuel_mass = 5.0
        mass_init = 10.0
        fuel_consumption_rate = 0.01

    #TanYingqi:记录奖励情况，并保存为csv文件  #SunYunru:加入版本控制
    #SunYunru:设置表头
    VERSION_HEADERS = {
        '_raw': ['episode', 'total_reward', 'dist_reward', 'pose_reward', 
                'landing_bonus', 'crash_penalty', 'step_id', 'landed', 'crashed'],
        '_wind&fuel': ['episode', 'total_reward', 'dist_reward', 'pose_reward',
                    'fuel_bonus', 'landing_bonus', 'crash_penalty', 
                    'fuel_left', 'step_id', 'landed', 'crashed']
    }
    headers = VERSION_HEADERS.get(version, VERSION_HEADERS['_raw'])
    log_path = os.path.join(ckpt_folder, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    #SunYunru:环境初始化  #TanYingqi:增加风力、燃料消耗影响
    env = Rocket(task=task, max_steps=max_steps, rocket_type=rocket_type,
             wind_enabled=wind_enabled,
             wind_force_max=wind_force_max,
             fuel_mass=fuel_mass,
             mass_init=mass_init,
             fuel_consumption_rate=fuel_consumption_rate,
             version=version)  #SunYunru:版本控制

    #SunYunru:策略网络初始化
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims, layer_norm=layer_norm, entropy_set=entropy_set).to(device)

    #SunYunru:加载上次训练的模型
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1], weights_only=False)  #SunYunru:兼容不同版本的pytorch
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']

    #SunYunru:正式训练
    for episode_id in range(last_episode_id, max_m_episode):
        if episode_id % 999 == 1 and record_video:  #SunYunru:设置视频保存功能，每1000轮训练保存一次视频
            video_path = os.path.join(ckpt_folder, f'train_ep_{episode_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 2/env.dt, (env.viewport_w, env.viewport_h))
        # training loop
        state = env.reset()
        rewards, log_probs, values, masks = [], [], [], []
        #SunYunru:每次训练的每一步
        for step_id in range(max_steps):
            action, log_prob, value = net.get_action(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)
            if episode_id % 999 == 1 and record_video:
                frame_0, frame_1 = env.render()
                video_writer.write(cv2.cvtColor(frame_0, cv2.COLOR_RGB2BGR))
                video_writer.write(cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR))
            elif episode_id % 449 == 1:
                env.render()
            if done or step_id == max_steps-1:
                _, _, Qval = net.get_action(state)
                net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=0.999)
                break
        if episode_id % 999 == 1 and record_video and 'video_writer' in locals():
            video_writer.release()

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))
        
        #TanYingqi:用csv记录奖励  #SunYunru:加入版本控制、稍作优化
        reward_parts = getattr(env, '_last_reward_parts', {})
        BOOL_HEADERS = {'landed', 'crashed'}
        row = [episode_id] + [reward_parts.get(field, False if field in BOOL_HEADERS else 0) for field in headers[1:]]
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

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



