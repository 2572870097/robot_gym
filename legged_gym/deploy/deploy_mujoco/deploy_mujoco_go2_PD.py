import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from threading import Thread
import pygame
import math

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
joystick_use = True
joystick_opened = False

if joystick_use:
    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
        print("✅ 手柄连接成功")
    except Exception as e:
        print(f"❌ 无法打开手柄：{e}")
        joystick_use = False
    # joystick thread exit flag
    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd
        
        while not exit_flag:
            # get joystick input
            pygame.event.get()
            # update robot command
            x_vel_cmd = -joystick.get_axis(1) * 1.0
            y_vel_cmd = -joystick.get_axis(0) * 1.0
            yaw_vel_cmd = -joystick.get_axis(3) * 1.0
            pygame.time.delay(50)

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.daemon = True
        joystick_thread.start()

def handle_keyboard_input(viewer):
    """Handle keyboard input for robot control"""
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    
    # Get keyboard input
    if viewer.is_key_down(ord('w')):
        x_vel_cmd = 0.5
    elif viewer.is_key_down(ord('s')):
        x_vel_cmd = -0.5
    else:
        x_vel_cmd = 0.0
        
    if viewer.is_key_down(ord('a')):
        y_vel_cmd = 0.5
    elif viewer.is_key_down(ord('d')):
        y_vel_cmd = -0.5
    else:
        y_vel_cmd = 0.0
        
    if viewer.is_key_down(ord('q')):
        yaw_vel_cmd = 1.0
    elif viewer.is_key_down(ord('e')):
        yaw_vel_cmd = -1.0
    else:
        yaw_vel_cmd = 0.0

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

# 在循环外定义步态参数
gait_frequency = 2.0  # 步频 (Hz)
stride_length = 0.15   # 基础步幅 (m)
max_hip_angle = 0.3    # 髋关节最大偏移 (rad)
max_knee_angle = 0.8   # 膝关节最大偏移 (rad)

def generate_simple_action(phase, cmd):
    """直接从命令生成简单动作，更强的运动响应"""
    vx, vy, vyaw = cmd
    
    # 检查是否有运动指令
    cmd_magnitude = abs(vx) + abs(vy) + abs(vyaw)
    
    # 如果没有指令，返回零动作（保持静止）
    if cmd_magnitude < 0.01:
        return np.zeros(12)
    
    # 增大动作幅度，确保能看到明显运动
    hip_amplitude = 2.0    # 进一步增大髋关节幅度
    thigh_amplitude = 1.5  # 增大大腿关节幅度
    calf_amplitude = 1.0   # 增大小腿关节幅度
    
    # 简化动作生成：直接映射到关节
    # 前向运动：前后腿交替
    forward_front = vx * hip_amplitude * math.sin(phase)
    forward_rear = vx * hip_amplitude * math.sin(phase + math.pi)
    
    # 转向运动：左右腿相反
    turn_left = vyaw * hip_amplitude * 0.8
    turn_right = -vyaw * hip_amplitude * 0.8
    
    # 侧向运动
    side_motion = vy * hip_amplitude * 0.6
    
    # 抬腿运动：只在有运动指令时进行
    if cmd_magnitude > 0.1:
        lift_motion = thigh_amplitude * 0.4 * math.sin(phase * 3)  # 提高频率
        knee_motion = -calf_amplitude * 0.6 * (1 + 0.3 * math.sin(phase * 2))
    else:
        lift_motion = 0
        knee_motion = 0
    
    # 生成12个关节的动作 (Go2: FL, FR, RL, RR × 3关节each)
    action = np.array([
        # FL (前左)
        forward_front + turn_left + side_motion,    # Hip
        lift_motion,                                # Thigh  
        knee_motion,                               # Calf
        
        # FR (前右)
        forward_front + turn_right - side_motion,   # Hip
        lift_motion,                                # Thigh
        knee_motion,                               # Calf
        
        # RL (后左)
        forward_rear + turn_left + side_motion,     # Hip
        lift_motion,                                # Thigh
        knee_motion,                               # Calf
        
        # RR (后右)
        forward_rear + turn_right - side_motion,    # Hip
        lift_motion,                                # Thigh
        knee_motion                                # Calf
    ])
    
    return action

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser(description="Go2简化PD控制器")
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    
    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # PD控制
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            
            # 步进仿真
            mujoco.mj_step(m, d)

            # 处理键盘输入（如果没有手柄）
            if not joystick_use:
                handle_keyboard_input(viewer)

            # 更新命令
            cmd[0] = x_vel_cmd
            cmd[1] = y_vel_cmd  
            cmd[2] = yaw_vel_cmd

            counter += 1
            if counter % control_decimation == 0:
                # 创建观测
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                linear_velocity = d.qvel[:3]
                omega = d.qvel[3:6]

                # 标准化观测
                qj_norm = (qj - default_angles) * dof_pos_scale
                dqj_norm = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega_norm = omega * ang_vel_scale
                
                # 计算当前步态相位 (0~2π)
                phase = 2 * math.pi * (time.time() - start) * gait_frequency
                
                # 直接从命令生成动作
                action = generate_simple_action(phase, cmd)
                
                # 减弱补偿项，避免干扰主要动作
                # 重力补偿：根据机身俯仰角调整 (减小影响)
                pitch_angle = gravity_orientation[1]  # 俯仰角
                knee_compensation = 0.5 * pitch_angle  # 减小俯仰补偿
                action[1::3] += knee_compensation  # 所有大腿关节 (1,4,7,10)
                
                # 轻微的速度反馈补偿 (减小影响)
                vel_error = cmd[:2] - linear_velocity[:2]
                hip_comp = 0.02 * vel_error[0]  # 减小补偿强度
                action[[0,6]] += hip_comp       # 前后髋关节
                action[[3,9]] -= hip_comp       # 左右髋关节
                
                # 计算目标关节位置
                target_dof_pos = default_angles + action * action_scale

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Cleanup
    if joystick_use and 'exit_flag' in globals():
        exit_flag = True
    print("🏁 程序结束")
