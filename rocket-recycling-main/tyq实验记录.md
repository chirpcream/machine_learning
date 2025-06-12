# 机器学习大作业

## git实现多人编程

使用工具：git、VScode

【给傻子的Git教程】https://www.bilibili.com/video/BV1Hkr7YYEh8?vd_source=23274d00140aafc65734bc29f0c6864b

【和傻子一起写代码】https://www.bilibili.com/video/BV1udEuzrEa7?vd_source=23274d00140aafc65734bc29f0c6864b

[如何使用 Git 进行多人协作开发（全流程图解）_git多人协作开发流程-CSDN博客](https://blog.csdn.net/whc18858/article/details/133209975)

## 模拟风力扰动

加在 `ax` 上一个扰动项：`ax += wind_force / mass`

## 模拟燃料消耗

- 每次喷气时减少燃料；
- 质量逐渐减小，影响加速度；
- 如剩余燃料越多，奖励越高。

##### 添加风力参数和初始燃料

```python
self.wind_enabled = True
self.wind_force_max = 3.0  # 单位 N，最大横向风力

self.mass_init = 100.0     # 火箭总质量（可调整）
self.fuel_mass = 90.0      # 可燃烧燃料
self.fuel_consumption_rate = 0.02  # 每次推力所耗 kg
```

##### 加入风力扰动和质量影响

```python
# 计算当前质量
mass = self.mass_init - self.fuel_mass
mass = max(mass, 10.0)  # 防止质量为负

# 风力扰动
wind_force = 0.0
if self.wind_enabled:
    wind_force = np.random.uniform(-self.wind_force_max, self.wind_force_max)
self._last_wind_force = wind_force  # 保存当前风速，用于绘图

ax = (fx + wind_force - rho*vx) / mass
ay = (fy - self.g - rho*vy) / mass

```

##### 加入燃料消耗

$$
Δm=\dot{m}=α⋅f/g
$$

$$
m_{fuel}(t+Δt)=max(0, m_{fuel}(t)−\dot{m})
$$

$f$：当前推力（单位 N）

$g$：重力加速度（约 9.8 m/s²）

$\alpha$：燃料消耗速率因子（单位 kg/“重力单位推力”） 

$\dot{m}$：当前时间步的燃料消耗量

$m_{\text{fuel}}$：剩余燃料质量

**推力越大，燃烧速度越快；**

推力以“g”为单位标准化（使其与火箭本身抗重力能力相关）；

```python
# 推力消耗燃料
if f > 0:
    self.fuel_mass -= self.fuel_consumption_rate * (f / self.g)  # 简单按推力归一化计算
    self.fuel_mass = max(self.fuel_mass, 0)

```

##### 将剩余燃料加入 reward

```python
if self.task == 'landing' and self.already_landing:
    reward += 0.1 * (self.fuel_mass / 30.0)
```

##### 燃料耗尽判失败

```python
if self.fuel_mass <= 0 and not self.already_landing:
    self.already_crash = True
```

##### 状态向量扩展：加入 fuel_ratio 与 step_ratio

`flatten()` 函数：

```python
x = np.array([...]) / 100.
fuel_ratio = np.array([self.fuel_mass / self.mass_init], dtype=np.float32)
step_ratio = np.array([state['t'] / self.max_steps], dtype=np.float32)
return np.concatenate([x, fuel_ratio, step_ratio])

```

在 `__init__()` 结尾设置：

```python
self.state_dims = 10  # 原为8，现在加入两个额外维度
```

##### 图像界面实时显示风速与燃料

`draw_text()` 函数中末尾加入：

```python
pt = (10, 120)
text = "fuel left: %.2f kg" % self.fuel_mass
put_text(canvas, text, pt)

pt = (10, 140)
if self.wind_enabled:
    text = "wind force: %.2f N" % self._last_wind_force
else:
    text = "wind force: OFF"
put_text(canvas, text, pt)

```

##### Rocket 初始化方式更新

```python
env = Rocket(task='hover', max_steps=800, rocket_type='falcon',
             wind_enabled=True,
             wind_force_max=2.5,
             fuel_mass=120.0,
             mass_init=140.0,
             fuel_consumption_rate=0.02)

```

##### 让转动惯量随质量变化

$$
I= \frac{1}{12}⋅m(t)⋅H^2
$$

$I$：火箭绕中心轴的转动惯量（单位 kg·m²）

$m(t)$：当前火箭总质量，随燃料减少而减小

$H$：火箭高度

```python
mass = max(self.mass_init - self.fuel_mass, 10.0)
I = (1/12) * mass * (self.H ** 2) 
atheta = ft * self.H/2 / I 
```

##### 非对称风力作用（风引起转动）

当前模型默认火箭为质量均匀的竖直矩形刚体，质心在几何中心（重心）处，即火箭中点、高度 $H/2$ 位置。

设定风的施力点相对于质心的偏移为：
$$
h_{\text{wind}} \sim \mathcal{U}(-H/2, H/2)
$$
我们希望风力不仅推动火箭平移，也能吹歪火箭，引发转动（角加速度）
$$
τ_{wind}=F_{wind}\cdot h_{\text{wind}}
$$
则
$$
\alpha_{\theta,wind}=\frac{\tau_{wind} }{I}
$$

```python
        mass = max(self.mass_init - self.fuel_mass, 10.0)
        I = (1/12) * mass * (self.H ** 2)  # 更新转动惯量

        tau_engine = ft * self.H/2 # 计算推力产生的角加速度
        # 引入风力随机扰动点位
        self.h_wind = np.random.uniform(-self.H/2, self.H/2)
        tau_wind = wind_force * self.h_wind
        atheta = (tau_engine + tau_wind) / I
```

在 `draw_text()` 中增加：

```python
pt = (10, 180)
put_text(canvas, "wind @ h = %.1f m" % h_wind, pt)

```

训练之后第一次reward比之前好了很多，我不是很懂，但是gpt这么说

<img src="tyq实验记录.assets/image-20250613051457553.png" alt="image-20250613051457553" style="zoom:50%;" />

<img src="tyq实验记录.assets/image-20250613051435392.png" alt="image-20250613051435392" style="zoom:50%;" />

<img src="tyq实验记录.assets/image-20250613051520902.png" alt="image-20250613051520902" style="zoom:50%;" />

##### 解决问题：每轮开始时火箭“油量是上轮剩下的”

```python
def __init__(...):
    ...
    self.fuel_mass_init = fuel_mass  # <--- 记录初始燃料
    self.fuel_mass = fuel_mass
    ...
```

reset()添加

```python
self.fuel_mass = self.fuel_mass_init 
```

这样就不会燃料突然消失然后非常吓人了

![image-20250613054034752](tyq实验记录.assets/image-20250613054034752.png)

## policy.py

原来的policy代码保存在副本里了

##### Entropy Loss 

鼓励策略在训练初期保持对动作的多样性探索。强化学习常常面临“早收敛”的问题，策略在尚未充分尝试所有可能动作之前就锁定在某个次优策略上，导致泛化能力差。通过对策略输出的动作分布计算熵值，并在损失函数中给予一定权重的正向奖励，可以有效防止策略过早变得过于保守，使其在面对复杂环境扰动（如风力、燃料变化）时仍具备探索能力，从而学到更稳健的控制策略。

```python
entropy = -(log_probs * torch.exp(log_probs)).sum()
actor_loss = (-log_probs * advantage.detach()).mean() - 0.001 * entropy
```

##### Layer Normalization（层归一化）

提升训练过程的稳定性

```python
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mapping = PositionalMapping(input_dim=input_dim, L=7)

        h_dim = 128
        # tyq
        self.linear1 = nn.Linear(self.mapping.output_dim, h_dim)
        self.norm1 = nn.LayerNorm(h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)
        self.norm2 = nn.LayerNorm(h_dim)
        self.linear3 = nn.Linear(h_dim, h_dim)
        self.norm3 = nn.LayerNorm(h_dim)
        self.linear4 = nn.Linear(h_dim, output_dim)
        self.relu = nn.LeakyReLU(0.2)

        # self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=h_dim, bias=True)
        # self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        # self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        # self.linear4 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        # self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # shape x: 1 x m_token x m_state
        # x = x.view([1, -1])
        # x = self.mapping(x)
        # x = self.relu(self.linear1(x))
        # x = self.relu(self.linear2(x))
        # x = self.relu(self.linear3(x))
        # x = self.linear4(x)
        # tyq
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.relu(self.norm1(self.linear1(x)))
        x = self.relu(self.norm2(self.linear2(x)))
        x = self.relu(self.norm3(self.linear3(x)))
        x = self.linear4(x)
        return x
```

## “推力变化惯性”机制

模拟现实中火箭发动机推力不是瞬时切换的，而是有惯性，改变推力时会渐进调整。
$$
f_{t+1}=\beta \cdot f_{t0} +(1- \beta) \cdot f_{target},β∈[0.8,0.98]
$$
$f_{\text{target}}$：策略当前选择的推力

$f_t$：当前真实推力值

$\beta \in [0.8, 0.98]$：惯性权重

```python
        # tyq 推力惯性
        # f, vphi = self.action_table[action]
        f_target, vphi = self.action_table[action]

        # 推力惯性参数
        self._throttle_beta = 0.9 if not hasattr(self, '_throttle_beta') else self._throttle_beta
        self.f = self.f if hasattr(self, 'f') else f_target  # 初始化上次推力

        # 平滑更新推力（模拟推力惯性）
        self.f = self._throttle_beta * self.f + (1 - self._throttle_beta) * f_target
        f = self.f
```

