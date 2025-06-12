import torch
from rocket import Rocket
from policy import ActorCritic
import os
import glob
import cv2

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'hover'  # 'hover' or 'landing'
    rocket_type = 'falcon'  #SunYunru:考虑变量rocket_type:可以选'falcon'或'starship'
    version = '_raw'  #SunYunru:增设变量version，方便对比不同修改下代码运行结果
    record_video = True  #SunYunru:增设变量record_video，确定是否保存视频
    max_steps = 800
    ckpt_folder = os.path.join('./', task + version + '_' + rocket_type + '_ckpt')
    ckpt_dir = glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1]  # last ckpt

    # tyq
    env = Rocket(task='hover', max_steps=800, rocket_type='falcon',
             wind_enabled=True,
             wind_force_max=2.5,
             fuel_mass=100.0,
             mass_init=120.0,
             fuel_consumption_rate=0.02)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir, weights_only=False)  #SunYunru:兼容不同版本的pytorch
        net.load_state_dict(checkpoint['model_G_state_dict'])

    if record_video:  #SunYunru:设置视频保存功能
        video_path = os.path.join(ckpt_folder, 'inference.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 2/env.dt, (env.viewport_w, env.viewport_h))
    state = env.reset()
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        if record_video:
            frame_0, frame_1 = env.render(window_name='test')
            video_writer.write(cv2.cvtColor(frame_0, cv2.COLOR_RGB2BGR))
            video_writer.write(cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR))
        if env.already_crash:
            break
    if record_video and 'video_writer' in locals():
        video_writer.release()