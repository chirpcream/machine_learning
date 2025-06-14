import torch
from rocket import Rocket
from policy import ActorCritic
import os
import glob
import cv2

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    #SunYunru:测试前必须注意的内容
    task = 'landing'  # 'hover' or 'landing'
    rocket_type = 'falcon'  #SunYunru:考虑变量rocket_type:可以选'falcon'或'starship'
    version = '_wind&fuel'  #SunYunru:增设变量version，方便对比不同修改下代码运行结果:可以选'_raw'或'_wind&fuel'
    entropy_set =True  #TanYingqi:增设变量entropy_set，促进策略多样性探索  #SunYunru:整合完善
    layer_norm = True  #TanYingqi:增设变量layer_norm，确定是否使用层归一化  #SunYunru:整合完善  #SunYunru:增加残差链接
    record_video = True  #SunYunru:增设变量record_video，确定是否保存视频
    max_steps = 800

    #SunYunru:常规初始化
    ckpt_folder = os.path.join('./', task + version + '_' + rocket_type + '_ckpt')
    ckpt_dir = glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1]  # last ckpt
    wind_enabled = False
    wind_force_max = 0.2
    fuel_mass = 3.0
    mass_init = 5.0
    fuel_consumption_rate = 0.005

    #SunYunru:版本控制
    if version == '_wind&fuel':  #SunYunru:考虑风力和燃料消耗影响时的版本控制
        wind_enabled = True
        wind_force_max = 0.2
        fuel_mass = 3.0
        mass_init = 5.0
        fuel_consumption_rate = 0.005


    #TanYingqi:增加风力、燃料消耗影响
    env = Rocket(task=task, max_steps=max_steps, rocket_type=rocket_type,
             wind_enabled=wind_enabled,
             wind_force_max=wind_force_max,
             fuel_mass=fuel_mass,
             mass_init=mass_init,
             fuel_consumption_rate=fuel_consumption_rate,
             version=version)  #SunYunru:版本控制
    
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims, layer_norm=layer_norm, entropy_set=entropy_set).to(device)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir, weights_only=False)  #SunYunru:兼容不同版本的pytorch
        net.load_state_dict(checkpoint['model_G_state_dict'])

    #SunYunru:初始化视频保存功能
    if record_video:  
        video_path = os.path.join(ckpt_folder, 'inference.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 2/env.dt, (env.viewport_w, env.viewport_h))

    #SunYunru:初始化状态
    state = env.reset()

    #SunYunru:开始测试
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        if record_video:
            frame_0, frame_1 = env.render(window_name='test')
            video_writer.write(cv2.cvtColor(frame_0, cv2.COLOR_RGB2BGR))
            video_writer.write(cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR))
        if env.already_crash:
            break

    #SunYunru:结束测试，释放视频资源
    if record_video and 'video_writer' in locals():
        video_writer.release()