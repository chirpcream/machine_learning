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
self.fuel_mass = 30.0      # 可燃烧燃料
self.fuel_consumption_rate = 0.1  # 每次推力所耗 kg
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

ax = (fx + wind_force - rho*vx) / mass
ay = (fy - self.g - rho*vy) / mass

```

##### 加入燃料消耗

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

