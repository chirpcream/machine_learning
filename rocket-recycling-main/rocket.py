import numpy as np
import random
import cv2
import utils


class Rocket(object):
    """
    Rocekt and environment.
    The rocket is simplified into a rigid body model with a thin rod,
    considering acceleration and angular acceleration and air resistance
    proportional to velocity.

    There are two tasks: hover and landing
    Their reward functions are straight forward and simple.

    For the hover tasks: the step-reward is given based on two factors
    1) the distance between the rocket and the predefined target point
    2) the angle of the rocket body (the rocket should stay as upright as possible)

    For the landing task: the step-reward is given based on three factors:
    1) the distance between the rocket and the predefined landing point.
    2) the angle of the rocket body (the rocket should stay as upright as possible)
    3) Speed and angle at the moment of contact with the ground, when the touching-speed
    are smaller than a safe threshold and the angle is close to 90 degrees (upright),
    we see it as a successful landing.

    """

    def __init__(self, max_steps, task='hover', rocket_type='falcon',
                 viewport_h=768, path_to_bg_img=None,
                 wind_enabled=True, wind_force_max=3,
                 mass_init=100.0, fuel_mass=90.0,
                 fuel_consumption_rate=0.02, version='_raw'):  #TanYingqi:增加风力、燃料消耗影响

        self.task = task
        self.rocket_type = rocket_type
        self.version = version  #SunYunru:版本控制

        self.g = 9.8
        self.H = 50  # rocket height (meters)
        self.I = 1/12*self.H*self.H  # Moment of inertia
        self.dt = 0.05

        self.world_x_min = -300  # meters
        self.world_x_max = 300
        self.world_y_min = -30
        self.world_y_max = 570

        # target point
        if self.task == 'hover':
            self.target_x, self.target_y, self.target_r = 0, 200, 50
        elif self.task == 'landing':
            self.target_x, self.target_y, self.target_r = 0, self.H/2.0, 50

        self.already_landing = False
        self.already_crash = False
        self.max_steps = max_steps

        # viewport height x width (pixels)
        self.viewport_h = int(viewport_h)
        self.viewport_w = int(viewport_h * (self.world_x_max-self.world_x_min) \
                          / (self.world_y_max - self.world_y_min))
        self.step_id = 0

        self.state = self.create_random_state()
        self.action_table = self.create_action_table()

        if version == '_raw':
            self.state_dims = 8
        elif version == '_wind&fuel':
            self.state_dims = 10  #TanYingqi:增加风力、燃料消耗影响
        self.action_dims = len(self.action_table)

        if path_to_bg_img is None:
            path_to_bg_img = task + '.jpg'
        self.bg_img = utils.load_bg_img(path_to_bg_img, w=self.viewport_w, h=self.viewport_h)

        self.state_buffer = []

        self.wind_enabled = wind_enabled
        if version == '_raw':
            self.wind_force_max = 0.0  #SunYunru:不考虑风力影响
        else:
            self.wind_force_max = wind_force_max  #TanYingqi:单位 N，最大横向风力
        self.mass_init = mass_init  #TanYingqi:火箭总质量
        self.fuel_mass_init = fuel_mass  #TanYingqi:初始可燃烧燃料
        self.fuel_mass = fuel_mass  #TanYingqi:可燃烧燃料
        self.structure_mass = self.mass_init - self.fuel_mass_init  #SunYunru:定义结构质量，优化代码表达
        self.fuel_consumption_rate = fuel_consumption_rate  #TanYingqi:每次推力所耗 kg
        self._last_wind_force = 0.0  #SunYunru:记录上一次风力




    def reset(self, state_dict=None):

        if state_dict is None:
            self.state = self.create_random_state()
        else:
            self.state = state_dict

        self.state_buffer = []
        self.step_id = 0
        self.already_landing = False
        self.fuel_mass = self.fuel_mass_init  #TanYingqi:每轮训练重置燃料质量
        cv2.destroyAllWindows()
        return self.flatten(self.state)

    def create_action_table(self):
        # f0 = 0.1 * self.g  # thrust
        # f1 = 1.0 * self.g
        # f2 = 2.2 * self.g
        # vphi0 = 0  # Nozzle angular velocity
        # vphi1 = 30 / 180 * np.pi
        # vphi2 = -30 / 180 * np.pi

        # action_table = [[f0, vphi0], [f0, vphi1], [f0, vphi2],
        #                 [f1, vphi0], [f1, vphi1], [f1, vphi2],
        #                 [f2, vphi0], [f2, vphi1], [f2, vphi2]
        #                 ]
        # return action_table

        #SunYunru:修改动作表
        f0 = 0.2 * self.g  # thrust
        f1 = 1.0 * self.g
        f2 = 2.0 * self.g
        f3 = 4.0 * self.g  #SunYunru:增加更大推力
        vphi0 = 0  # Nozzle angular velocity
        vphi1 = 30 / 180 * np.pi
        vphi2 = -30 / 180 * np.pi
        vphi3 = 60 / 180 * np.pi  #SunYunru:增加更大角速度
        vphi4 = -60 / 180 * np.pi

        action_table = [[f0, vphi0], [f0, vphi1], [f0, vphi2], [f0, vphi3], [f0, vphi4],
                        [f1, vphi0], [f1, vphi1], [f1, vphi2], [f1, vphi3], [f1, vphi4],
                        [f2, vphi0], [f2, vphi1], [f2, vphi2], [f2, vphi3], [f2, vphi4],
                        [f3, vphi0], [f3, vphi1], [f3, vphi2], [f3, vphi3], [f3, vphi4]
                        ]
        return action_table


    def get_random_action(self):
        return random.randint(0, len(self.action_table)-1)

    def create_random_state(self):

        # predefined locations
        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min
        xc = (self.world_x_max + self.world_x_min) / 2.0
        yc = (self.world_y_max + self.world_y_min) / 2.0

        if self.task == 'landing':
            x = random.uniform(xc - x_range / 4.0, xc + x_range / 4.0)
            y = yc + 0.4*y_range
            if x <= 0:
                theta = -85 / 180 * np.pi
            else:
                theta = 85 / 180 * np.pi
            vy = -50

        if self.task == 'hover':
            x = xc
            y = yc + 0.2 * y_range
            theta = random.uniform(-45, 45) / 180 * np.pi
            vy = -10

        state = {
            'x': x, 'y': y, 'vx': 0, 'vy': vy,
            'theta': theta, 'vtheta': 0,
            'phi': 0, 'f': 0,
            't': 0, 'a_': 0
        }

        return state

    def check_crash(self, state):
        if self.task == 'hover':
            x, y = state['x'], state['y']
            theta = state['theta']
            crash = False
            if y <= self.H / 2.0:
                crash = True
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            return crash

        elif self.task == 'landing':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = (vx**2 + vy**2)**0.5

            crash = False
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            if y <= 0 + self.H / 2.0 and v >= 15.0:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(x) >= self.target_r:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(theta) >= 10/180*np.pi:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(vtheta) >= 10/180*np.pi:
                crash = True
            return crash

    def check_landing_success(self, state):
        if self.task == 'hover':
            return False
        elif self.task == 'landing':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = (vx**2 + vy**2)**0.5
            return True if y <= 0 + self.H / 2.0 and v < 15.0 and abs(x) < self.target_r \
                           and abs(theta) < 10/180*np.pi and abs(vtheta) < 10/180*np.pi else False

    def calculate_reward(self, state):

        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min

        # dist between agent and target point
        dist_x = abs(state['x'] - self.target_x)
        dist_y = abs(state['y'] - self.target_y)
        dist_norm = dist_x / x_range + dist_y / y_range
        distance = (dist_x**2 + dist_y**2) ** 0.5  #TanYingqi:计算距离
        theta = abs(state['theta']) 
        v = (state['vx'] ** 2 + state['vy'] ** 2) ** 0.5

        # dist_reward = 0.1*(1.0 - dist_norm)
        dist_reward = 0.0
        pose_reward = 0.0
        fuel_bonus = 0.0
        precision_bonus = 0.0
        landing_bonus = 0.0
        crash_penalty = 0.0

        #SunYunru:奖励计算结构说明
        '''
        这里对奖励计算结构进行了较大的改动，解释如下：
        landing工况未坠毁、着陆，hover工况倾角不过大时，总奖励= 距离奖励（常规指数型距离奖励+附加距离奖励） + 悬停角度奖励 + 燃料节省奖励 + 精确悬停奖励
        landing工况坠毁，则总奖励为坠毁惩罚，目前设为了0
        landing工况着陆成功，则总奖励为按照特定公式计算的着陆奖励，加上燃料节省奖励和精确悬停奖励
        hover工况倾角过大，则总奖励为0
        其中：
        - 距离奖励：使用指数函数计算距离奖励，距离越近奖励越高，并且在距离小于10或20时增加附加奖励
        - 悬停角度奖励：如果姿态角度小于30度，则奖励0.1，否则根据角度大小计算奖励
        - 燃料节省奖励：根据剩余燃料质量计算奖励
        - 精确悬停奖励：如果长期保持在目标点附近且姿态角度小于阈值，则额外一次性增加奖励
        - 着陆奖励：如果成功着陆，则奖励为1.0 + 5 * exp(-v / 10.0) * (剩余步数)，其中v为速度
        - 坠毁惩罚：如果已经坠毁，则奖励为坠毁惩罚
        注意：如果燃料质量为0，则认为坠毁
        以上奖励计算方式旨在鼓励火箭在悬停和着陆任务中尽量减少距离目标点的偏差，保持正确的姿态，并节省燃料，同时避免坠毁。
        其中，距离奖励和姿态奖励是常规的指数型奖励，附加距离奖励和精确悬停奖励则是为了鼓励火箭在目标点附近保持稳定悬停。
        '''
        #TanYingqi:常规指数型距离奖励
        dist_reward = np.exp(-distance / 30.0) * 0.5

        #TanYingqi:附加距离奖励
        if distance < 10.0:
            dist_reward += 0.2
        elif distance < 20.0:
            dist_reward += 0.1

        #TanYingqi:悬停角度奖励
        if abs(state['theta']) <= np.pi / 6.0:
            pose_reward = 0.1
        else:
            pose_reward = abs(state['theta']) / (0.5*np.pi)
            pose_reward = 0.1 * (1.0 - pose_reward)

        #TanYingqi:燃料节省奖励
        fuel_bonus = 0.03 * (self.fuel_mass / 30.0)

        #TanYingqi:精确悬停奖励（持续计算点数，达到阈值一次性触发）
        if not hasattr(self, 'precision_hover_counter'):
            self.precision_hover_counter = 0
            self.precision_hover_threshold = 10
        if distance < 5.0 and theta < 10 / 180 * np.pi:
            self.precision_hover_counter += 1
        else:
            self.precision_hover_counter = 0

        if self.precision_hover_counter >= self.precision_hover_threshold:
            if self.task == 'hover':
                precision_bonus = 10.0  
            elif self.task == 'landing':
                precision_bonus = 2.0   
            self.precision_hover_counter = 0

        #SunYunru:坠毁和着陆奖励
        if self.task == 'landing':
            if self.already_crash:
                crash_penalty = (dist_reward + pose_reward + 5 * np.exp(-v / 10.0)) * (self.max_steps - self.step_id)
            elif self.already_landing:
                landing_bonus = (1.0 + 5 * np.exp(-v / 10.0)) * (self.max_steps - self.step_id)

        #SunYunru:奖励计算
        if self.task == 'landing':
            if self.already_crash:
                reward = crash_penalty
            elif self.already_landing:
                reward = landing_bonus + fuel_bonus + precision_bonus
            else:
                reward = dist_reward + pose_reward + fuel_bonus + precision_bonus
        elif self.task == 'hover':
            if theta > 90 / 180 * np.pi:
                reward = 0.0  #TanYingqi:姿态过大直接判定失败
            else:
                reward = dist_reward + pose_reward + fuel_bonus + precision_bonus

        #Tanyingqi:奖励分解并记录
        self._last_reward_parts = {
        'dist_reward': float(dist_reward),
        'pose_reward': float(pose_reward),
        'fuel_bonus': float(fuel_bonus),
        'landing_bonus': float(landing_bonus),
        'crash_penalty': float(crash_penalty),
        'total_reward': float(reward),
        'fuel_left': float(self.fuel_mass),
        'step_id': self.step_id,
        'landed': self.already_landing,
        'crashed': self.already_crash
        }            

        return reward

    def step(self, action):
        #SunYunru:有关力计算的说明
        '''
        这里火箭主要受到推力、重力、风力的影响，风力的计算方法目前比较脱离实际,考虑到这不是研究的重点，故先用这个方法计算。
        这里的推力和风力都做了一定的平滑处理。
        '''
        x, y, vx, vy = self.state['x'], self.state['y'], self.state['vx'], self.state['vy']
        theta, vtheta = self.state['theta'], self.state['vtheta']
        phi = self.state['phi']

        #TanYingqi:推力惯性
        # f, vphi = self.action_table[action]
        f_target, vphi = self.action_table[action]

        #TanYingqi:推力惯性参数
        self._throttle_beta = 0.05 if not hasattr(self, '_throttle_beta') else self._throttle_beta  #SunYunru:降低推力惯性参数
        self.f = self.f if hasattr(self, 'f') else f_target  #TanYingqi:初始化上次推力

        #TanYingqi:平滑更新推力（模拟推力惯性）
        self.f = self._throttle_beta * self.f + (1 - self._throttle_beta) * f_target
        f = self.f

        #TanYingqi:推力消耗燃料
        if f > 0:
            if self.version == '_wind&fuel':
                self.fuel_mass -= self.fuel_consumption_rate * (f / self.g)  #TanYingqi:简单按推力归一化计算
                self.fuel_mass = max(self.fuel_mass, 0)  #TanYingqi:避免为负
            elif self.version == '_raw':
                self.fuel_mass = self.fuel_mass_init

        ft, fr = -f*np.sin(phi), f*np.cos(phi)
        fx = ft*np.cos(theta) - fr*np.sin(theta)
        fy = ft*np.sin(theta) + fr*np.cos(theta)

        rho = 1 / (125/(self.g/2.0))**0.5  # suppose after 125 m free fall, then air resistance = mg
        
        #TanYingqi:更新质量和风力影响 #SunYunru:修改物理逻辑错误
        if self.version == '_wind&fuel':
            mass = self.structure_mass + self.fuel_mass #SunYunru:定义结构质量，优化代码表达
        elif self.version == '_raw':
            mass = self.mass_init

        wind_force = 0.0
        if self.wind_enabled:
            target_wind =np.random.uniform(-self.wind_force_max, self.wind_force_max)
            wind_force = self._last_wind_force * 0.9 + target_wind * 0.1  #SunYunru:平滑更新风力，避免风向突变的脱离实际的情况
        self._last_wind_force = wind_force

        ax = (fx * (self.structure_mass + self.fuel_mass_init) + wind_force - rho * vx) / mass
        ay = (fy * (self.structure_mass + self.fuel_mass_init) - self.g * mass - rho * vy) / mass

        #TanYingqi:更新转动惯量
        I = (1/12) * mass * (self.H ** 2)  #TanYingqi:更新转动惯量 #SunYunru:这里转动惯量的计算应该进行了简化，认为质量是均匀分布的

        tau_engine = ft * self.H/2 #TanYingqi:计算推力产生的角加速度
        #TanYingqi:引入风力随机扰动点位
        if not hasattr(self, 'h_wind') or self.step_id % 10 == 0:  #SunYunru:每10步更新一次        
            self.h_wind = np.random.uniform(-self.H/2, self.H/2)
        tau_wind = wind_force * self.h_wind
        atheta = (tau_engine + tau_wind) / I

        # update agent
        if self.already_landing:
            vx, vy, ax, ay, theta, vtheta, atheta = 0, 0, 0, 0, 0, 0, 0
            phi, f = 0, 0
            action = 0

        self.step_id += 1
        x_new = x + vx*self.dt + 0.5 * ax * (self.dt**2)
        y_new = y + vy*self.dt + 0.5 * ay * (self.dt**2)
        vx_new, vy_new = vx + ax * self.dt, vy + ay * self.dt
        theta_new = theta + vtheta*self.dt + 0.5 * atheta * (self.dt**2)
        vtheta_new = vtheta + atheta * self.dt
        phi = phi + self.dt*vphi

        phi = max(phi, -20/180*3.1415926)
        phi = min(phi, 20/180*3.1415926)

        self.state = {
            'x': x_new, 'y': y_new, 'vx': vx_new, 'vy': vy_new,
            'theta': theta_new, 'vtheta': vtheta_new,
            'phi': phi, 'f': f,
            't': self.step_id, 'action_': action
        }
        self.state_buffer.append(self.state)

        self.already_landing = self.check_landing_success(self.state)
        self.already_crash = self.check_crash(self.state)
        reward = self.calculate_reward(self.state)

        #TanYingqi:如果燃料为0认为坠毁
        if self.fuel_mass <= 0 and not self.already_landing:
            self.already_crash = True  

        if self.already_crash or self.already_landing:
            done = True
        else:
            done = False

        return self.flatten(self.state), reward, done, None

    def flatten(self, state):
        x = [state['x'], state['y'], state['vx'], state['vy'],
             state['theta'], state['vtheta'], state['t'],
             state['phi']]
        #TanYingqi:加入燃料质量与步数归一化
        x = np.array(x, dtype=np.float32)/100.

        if self.version == '_raw':  #SunYunru
            return x
        elif self.version == '_wind&fuel':
            fuel_ratio = np.array([self.fuel_mass / self.mass_init], dtype=np.float32)
            step_ratio = np.array([state['t'] / self.max_steps], dtype=np.float32)
            return np.concatenate([x, fuel_ratio, step_ratio])

    def render(self, window_name='env', wait_time=1,
               with_trajectory=True, with_camera_tracking=True,
               crop_scale=0.4):

        canvas = np.copy(self.bg_img)
        polys = self.create_polygons()

        # draw target region
        for poly in polys['target_region']:
            self.draw_a_polygon(canvas, poly)
        # draw rocket
        for poly in polys['rocket']:
            self.draw_a_polygon(canvas, poly)
        frame_0 = canvas.copy()

        # draw engine work
        for poly in polys['engine_work']:
            self.draw_a_polygon(canvas, poly)
        frame_1 = canvas.copy()

        if with_camera_tracking:
            frame_0 = self.crop_alongwith_camera(frame_0, crop_scale=crop_scale)
            frame_1 = self.crop_alongwith_camera(frame_1, crop_scale=crop_scale)

        # draw trajectory
        if with_trajectory:
            self.draw_trajectory(frame_0)
            self.draw_trajectory(frame_1)

        # draw text
        self.draw_text(frame_0, color=(0, 0, 0))
        self.draw_text(frame_1, color=(0, 0, 0))

        cv2.imshow(window_name, frame_0[:,:,::-1])
        cv2.waitKey(wait_time)
        cv2.imshow(window_name, frame_1[:,:,::-1])
        cv2.waitKey(wait_time)
        return frame_0, frame_1

    def create_polygons(self):

        polys = {'rocket': [], 'engine_work': [], 'target_region': []}

        if self.rocket_type == 'falcon':

            H, W = self.H, self.H/10
            dl = self.H / 30

            # rocket main body
            pts = [[-W/2, H/2], [W/2, H/2], [W/2, -H/2], [-W/2, -H/2]]
            polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})
            # rocket paint
            pts = utils.create_rectangle_poly(center=(0, -0.35*H), w=W, h=0.1*H)
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
            pts = utils.create_rectangle_poly(center=(0, -0.46*H), w=W, h=0.02*H)
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
            # rocket landing rack
            pts = [[-W/2, -H/2], [-W/2-H/10, -H/2-H/20], [-W/2, -H/2+H/20]]
            polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})
            pts = [[W/2, -H/2], [W/2+H/10, -H/2-H/20], [W/2, -H/2+H/20]]
            polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})

        elif self.rocket_type == 'starship':

            H, W = self.H, self.H / 2.6
            dl = self.H / 30

            # rocket main body (right half)
            pts = np.array([[ 0.        ,  0.5006878 ],
                           [ 0.03125   ,  0.49243465],
                           [ 0.0625    ,  0.48143053],
                           [ 0.11458334,  0.43878955],
                           [ 0.15277778,  0.3933975 ],
                           [ 0.2326389 ,  0.23796424],
                           [ 0.2326389 , -0.49931225],
                           [ 0.        , -0.49931225]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})

            # rocket main body (left half)
            pts = np.array([[-0.        ,  0.5006878 ],
                           [-0.03125   ,  0.49243465],
                           [-0.0625    ,  0.48143053],
                           [-0.11458334,  0.43878955],
                           [-0.15277778,  0.3933975 ],
                           [-0.2326389 ,  0.23796424],
                           [-0.2326389 , -0.49931225],
                           [-0.        , -0.49931225]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (212, 212, 232), 'edge_color': None})

            # upper wing (right)
            pts = np.array([[0.15972222, 0.3933975 ],
                           [0.3784722 , 0.303989  ],
                           [0.3784722 , 0.2352132 ],
                           [0.22916667, 0.23658872]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

            # upper wing (left)
            pts = np.array([[-0.15972222,  0.3933975 ],
                           [-0.3784722 ,  0.303989  ],
                           [-0.3784722 ,  0.2352132 ],
                           [-0.22916667,  0.23658872]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

            # lower wing (right)
            pts = np.array([[ 0.2326389 , -0.16368638],
                           [ 0.4548611 , -0.33562586],
                           [ 0.4548611 , -0.48555708],
                           [ 0.2638889 , -0.48555708]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

            # lower wing (left)
            pts = np.array([[-0.2326389 , -0.16368638],
                           [-0.4548611 , -0.33562586],
                           [-0.4548611 , -0.48555708],
                           [-0.2638889 , -0.48555708]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

        else:
            raise NotImplementedError('rocket type [%s] is not found, please choose one '
                                      'from (falcon, starship)' % self.rocket_type)

        # engine work
        f, phi = self.state['f'], self.state['phi']
        c, s = np.cos(phi), np.sin(phi)

        if f > 0 and f < 0.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
        elif f > 0.5 * self.g and f < 1.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
        elif f > 1.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
            pts4 = utils.create_rectangle_poly(center=(12 * dl * s, -H / 2 - 12 * dl * c), w=3 * dl, h=3 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts4, 'face_color': (255, 255, 255), 'edge_color': None})
        # target region
        if self.task == 'hover':
            pts1 = utils.create_rectangle_poly(center=(self.target_x, self.target_y), w=0, h=self.target_r/3.0)
            pts2 = utils.create_rectangle_poly(center=(self.target_x, self.target_y), w=self.target_r/3.0, h=0)
            polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
        else:
            pts1 = utils.create_ellipse_poly(center=(0, 0), rx=self.target_r, ry=self.target_r/4.0)
            pts2 = utils.create_rectangle_poly(center=(0, 0), w=self.target_r/3.0, h=0)
            pts3 = utils.create_rectangle_poly(center=(0, 0), w=0, h=self.target_r/6.0)
            polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts3, 'face_color': None, 'edge_color': (242, 242, 242)})

        # apply transformation
        for poly in polys['rocket'] + polys['engine_work']:
            M = utils.create_pose_matrix(tx=self.state['x'], ty=self.state['y'], rz=self.state['theta'])
            pts = np.array(poly['pts'])
            pts = np.concatenate([pts, np.ones_like(pts)], axis=-1)  # attach z=1, w=1
            pts = np.matmul(M, pts.T).T
            poly['pts'] = pts[:, 0:2]

        return polys


    def draw_a_polygon(self, canvas, poly):

        pts, face_color, edge_color = poly['pts'], poly['face_color'], poly['edge_color']
        pts_px = self.wd2pxl(pts)
        if face_color is not None:
            cv2.fillPoly(canvas, [pts_px], color=face_color, lineType=cv2.LINE_AA)
        if edge_color is not None:
            cv2.polylines(canvas, [pts_px], isClosed=True, color=edge_color, thickness=1, lineType=cv2.LINE_AA)

        return canvas


    def wd2pxl(self, pts, to_int=True):

        pts_px = np.zeros_like(pts)

        scale = self.viewport_w / (self.world_x_max - self.world_x_min)
        for i in range(len(pts)):
            pt = pts[i]
            x_p = (pt[0] - self.world_x_min) * scale
            y_p = (pt[1] - self.world_y_min) * scale
            y_p = self.viewport_h - y_p
            pts_px[i] = [x_p, y_p]

        if to_int:
            return pts_px.astype(int)
        else:
            return pts_px

    def draw_text(self, canvas, color=(255, 255, 0)):

        def put_text(vis, text, pt):
            cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)

        pt = (10, 20)
        text = "simulation time: %.2fs" % (self.step_id * self.dt)
        put_text(canvas, text, pt)

        pt = (10, 40)
        text = "simulation steps: %d" % (self.step_id)
        put_text(canvas, text, pt)

        pt = (10, 60)
        text = "x: %.2f m, y: %.2f m" % \
               (self.state['x'], self.state['y'])
        put_text(canvas, text, pt)

        pt = (10, 80)
        text = "vx: %.2f m/s, vy: %.2f m/s" % \
               (self.state['vx'], self.state['vy'])
        put_text(canvas, text, pt)

        pt = (10, 100)
        text = "a: %.2f degree, va: %.2f degree/s" % \
               (self.state['theta'] * 180 / np.pi, self.state['vtheta'] * 180 / np.pi)
        put_text(canvas, text, pt)

        if self.version == '_wind&fuel':
            #TanYingqi:绘制风力和燃料剩余量
            pt = (10, 120)
            text = "fuel left: %.2f kg" % self.fuel_mass
            put_text(canvas, text, pt)

            pt = (10, 140)
            if self.wind_enabled:
                text = "wind force: %.2f N" % self._last_wind_force  #TanYingqi:自定义属性
            else:
                text = "wind force: OFF"
            put_text(canvas, text, pt)
            pt = (10, 160)
            put_text(canvas, "wind_h = %.1f m" % self.h_wind, pt)  #TanYingqi:风吹的位置高度（相对质心）

            #TanYingqi:绘制平滑推力
            pt = (10, 180)
            put_text(canvas, "smoothed thrust: %.2f N" % self.f, pt)


    def draw_trajectory(self, canvas, color=(255, 0, 0)):

        pannel_w, pannel_h = 256, 256
        traj_pannel = 255 * np.ones([pannel_h, pannel_w, 3], dtype=np.uint8)

        sw, sh = pannel_w/self.viewport_w, pannel_h/self.viewport_h  # scale factors

        # draw horizon line
        range_x, range_y = self.world_x_max - self.world_x_min, self.world_y_max - self.world_y_min
        pts = [[self.world_x_min + range_x/3, self.H/2], [self.world_x_max - range_x/3, self.H/2]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # draw vertical line
        pts = [[0, self.H/2], [0, self.H/2+range_y/20]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        if len(self.state_buffer) < 2:
            return

        # draw traj
        pts = []
        for state in self.state_buffer:
            pts.append([state['x'], state['y']])
        pts_px = self.wd2pxl(pts)

        dn = 5
        for i in range(0, len(pts_px)-dn, dn):

            x1, y1 = int(pts_px[i][0]*sw), int(pts_px[i][1]*sh)
            x1_, y1_ = int(pts_px[i+dn][0]*sw), int(pts_px[i+dn][1]*sh)

            cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x1_, y1_), color=color, thickness=2, lineType=cv2.LINE_AA)

        roi_x1, roi_x2 = self.viewport_w - 10 - pannel_w, self.viewport_w - 10
        roi_y1, roi_y2 = 10, 10 + pannel_h
        canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] = 0.6*canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] + 0.4*traj_pannel



    def crop_alongwith_camera(self, vis, crop_scale=0.4):
        x, y = self.state['x'], self.state['y']
        xp, yp = self.wd2pxl([[x, y]])[0]
        crop_w_half, crop_h_half = int(self.viewport_w*crop_scale), int(self.viewport_h*crop_scale)
        # check boundary
        if xp <= crop_w_half + 1:
            xp = crop_w_half + 1
        if xp >= self.viewport_w - crop_w_half - 1:
            xp = self.viewport_w - crop_w_half - 1
        if yp <= crop_h_half + 1:
            yp = crop_h_half + 1
        if yp >= self.viewport_h - crop_h_half - 1:
            yp = self.viewport_h - crop_h_half - 1

        x1, x2, y1, y2 = xp-crop_w_half, xp+crop_w_half, yp-crop_h_half, yp+crop_h_half
        vis = vis[y1:y2, x1:x2, :]

        vis = cv2.resize(vis, (self.viewport_w, self.viewport_h))
        return vis