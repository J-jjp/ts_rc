import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
import onnxruntime as ort
from legged_gym.envs import *
import torch
from scipy.spatial.transform import Rotation as R
import rospy
from sensor_msgs.msg import Joy
import os

import time

default_dof_pos=[0.1,0.8,-1.5 ,-0.1,0.8,-1.5, 0.1,1,-1.5, -0.1,1,-1.5]#默认角度需要与isacc一致
torque_limits = np.array([20.0, 55.0, 55.0, 20.0, 55.0, 55.0, 20.0, 55.0, 55.0, 20.0, 55.0, 55.0])

joy_cmd = [0.0, 0.0, 0.0]
from pynput import keyboard
##############
# 初始化键盘监听
##############
# 全局变量，保存键盘状态
key_state = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
    "l": False,
    "r": False,
}
# 最大和最小速度限制
max_speed = 1.0
min_speed = -1.0
# 阻尼系数，用于速度平滑下降
damping_factor = 0.2  # 调整该值以改变速度衰减快慢




def on_press(key):
    """按键按下时的回调函数"""
    if key == keyboard.Key.up:
        key_state["up"] = True
    elif key == keyboard.Key.down:
        key_state["down"] = True
    elif key == keyboard.Key.left:
        key_state["left"] = True
    elif key == keyboard.Key.right:
        key_state["right"] = True
    elif key == keyboard.KeyCode.from_char("r"):
        key_state["l"] = True
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["r"] = True


def on_release(key):
    """按键释放时的回调函数"""
    if key == keyboard.Key.up:
        key_state["up"] = False
    elif key == keyboard.Key.down:
        key_state["down"] = False
    elif key == keyboard.Key.left:
        key_state["left"] = False
    elif key == keyboard.Key.right:
        key_state["right"] = False
    elif key == keyboard.KeyCode.from_char("r"):
        key_state["l"] = False
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["r"] = False
global command

#########################
# 监听键盘输入以发送 command
#########################
command = np.array([0.0, 0.0, 0.0])
prev_command = np.array([0.0, 0.0, 0.0])
print(
    "Use arrow keys to move the robot.\n",
    "'l': turn left\n",
    "'r': turn left\n",
    "UP: move forward,\n",
    "DOWN: move backward,\n",
    "LEFT: move left,\n" "RIGHT: move right.\n",
)
# 监听键盘输入
command_scale_factor = 0.005
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def joy_callback(joy_msg):
    global joy_cmd
    joy_cmd[0] =  joy_msg.axes[1]
    joy_cmd[1] =  joy_msg.axes[0]
    joy_cmd[2] =  joy_msg.axes[3]  # 横向操作

def quat_rotate_inverse(q, v):
    # 确保输入为numpy数组
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('imu_quat').data[[1, 2, 3, 0]].astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    vel = data.sensor('base_lin_vel').data.astype(np.double)
    return (q, dq, quat,omega,vel)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd



def normalize_l2_numpy(x, axis=-1, eps=1e-12):
    """L2 normalize input array along specified axis."""
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)  # 计算L2范数
    return x / np.clip(norm, a_min=eps, a_max=None)  # 防止除以零

# 示例用法



class Sim2simCfg(A1RoughCfg):

    class sim_config:
        # print("{LEGGED_GYM_ROOT_DIR}",{LEGGED_GYM_ROOT_DIR})

        mujoco_model_path = '/home/ubuntu/isaac/t_s/quadruped_rl_blind_walk/legged_gym/resources/robots/TOE_dog2/xml/scene.xml'
        # mujoco_model_path = "/home/ubuntu/isaac/t_s/quadruped_rl_blind_walk/mujoco_test/data/go2/scene_terrain.xml"
        
        sim_duration = 60.0
        dt = 0.005 #1Khz底层
        decimation = 4 # 50Hz

    class robot_config:

        kps = np.array(20, dtype=np.double)#PD和isacc内部一致
        kds = np.array(0.5, dtype=np.double)
        tau_limit = 30. * np.ones(12, dtype=np.double)#nm


if __name__ == '__main__':
    rospy.init_node('play')
    rospy.Subscriber('/joy', Joy, joy_callback, queue_size=10)


    encoder_model_path = "/home/ubuntu/isaac/t_s/quadruped_rl_blind_walk/onnx/encoder_z_input.onnx"
    policy_model_path = "/home/ubuntu/isaac/t_s/quadruped_rl_blind_walk/onnx/legged.onnx"

    
    encoder = ort.InferenceSession(encoder_model_path, 
                            providers=['CPUExecutionProvider'])
    policy = ort.InferenceSession(policy_model_path, 
                            providers=['CPUExecutionProvider'])
    model = mujoco.MjModel.from_xml_path(Sim2simCfg.sim_config.mujoco_model_path)#载入初始化位置由XML决定
    model.opt.timestep = Sim2simCfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    model.opt.gravity = (0, 0, -9.81) 
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((12), dtype=np.double)
    action = np.zeros((12), dtype=np.double)
    action_flt = np.zeros((12), dtype=np.double)
    last_actions = np.zeros((12), dtype=np.double)
    lag_buffer = [np.zeros_like(action) for i in range(2+1)]

    hist_obs = deque()
    for _ in range(15):
        hist_obs.append(np.zeros([1,45], dtype=np.double))
    count_lowlevel = 0

    for _ in tqdm(range(int(Sim2simCfg.sim_config.sim_duration*10/ Sim2simCfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat,omega,vel= get_obs(data)#从mujoco获取仿真数据
        print("vin",vel[0])
        q = q[-12:]
        dq = dq[-12:]
        # print("dq",dq[1])
        # 持续更新 command，按住键时加速或减速
        if key_state["up"]: command[0] = min(command[0] + command_scale_factor, max_speed)
        elif key_state["down"]: command[0] = max(command[0] - command_scale_factor, min_speed)
        else:
            # 平滑衰减 x 方向速度
            if command[0] > 0: command[0] = max(command[0] - damping_factor, 0)
            elif command[0] < 0: command[0] = min(command[0] + damping_factor, 0)

        if key_state["left"]: command[1] = min(command[1] + command_scale_factor, max_speed)
        elif key_state["right"]: command[1] = max(command[1] - command_scale_factor, min_speed)
        else:
            # 平滑衰减 y 方向速度
            if command[1] > 0: command[1] = max(command[1] - damping_factor, 0)
            elif command[1] < 0: command[1] = min(command[1] + damping_factor, 0)

        if key_state["l"]: command[2] = max(command[2] - command_scale_factor, -1.0)  # 限制角速度最小为 -1
        elif key_state["r"]: command[2] = min(command[2] + command_scale_factor, 1.0)  # 限制角速度最大为 1
        else:
            # 平滑衰减角速度
            if command[2] > 0: command[2] = max(command[2] - damping_factor, 0)
            elif command[2] < 0: command[2] = min(command[2] + damping_factor, 0)
        # obs_buf =torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,
        #                     self.base_euler_xyz * self.obs_scales.quat,
        #                     self.commands[:, :3] * self.commands_scale,#xy+航向角速度
        #                     self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
        #                     self.reindex(self.dof_vel * self.obs_scales.dof_vel),
        #                     self.action_history_buf[:,-1]),dim=-1)#列表最后一项 [:-1]也就是上一次的

        if 1:
            # 1000hz ->50hz
            if count_lowlevel % Sim2simCfg.sim_config.decimation == 0:

                obs = np.zeros([1, 45], dtype=np.float32) #1,45           
                gravity_vec =  np.array([0., 0., -1.], dtype=np.float32)

                proj_gravity = quat_rotate_inverse(quat,gravity_vec)

                obs[0, 0] = omega[0] *Sim2simCfg.normalization.obs_scales.ang_vel
                obs[0, 1] = omega[1] *Sim2simCfg.normalization.obs_scales.ang_vel
                obs[0, 2] = omega[2] *Sim2simCfg.normalization.obs_scales.ang_vel
                obs[0, 3] = proj_gravity[0] 
                obs[0, 4] = proj_gravity[1] 
                obs[0, 5] = proj_gravity[2] 
                # print("obs:",proj_gravity)
                # if joy_cmd[0]>0:
                #     joy_cmd[0] = joy_cmd[0]*3
                # else:
                #     joy_cmd[0] = joy_cmd[0]
                obs[0, 6] = (joy_cmd[0]+command[0])* Sim2simCfg.normalization.obs_scales.lin_vel*3.5
                obs[0, 7] = (joy_cmd[1] +command[1])* Sim2simCfg.normalization.obs_scales.lin_vel*0.6
                obs[0, 8] = (joy_cmd[2] +command[2])* Sim2simCfg.normalization.obs_scales.ang_vel*0.8*2
                obs[0, 9:21] = (q-default_dof_pos) * Sim2simCfg.normalization.obs_scales.dof_pos #g关节角度顺序依据修改为样机
                obs[0, 21:33] = dq * Sim2simCfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = last_actions#上次控制指令
                obs = np.clip(obs, -Sim2simCfg.normalization.clip_observations, Sim2simCfg.normalization.clip_observations)

                # obs_cpu = obs  # 首先将Tensor移动到CPU，然后转换为NumPy数组 
                # for i in range(3):
                #     print("{:.2f}".format(obs_cpu[0][i]))
                # for i in range(3):  
                #     print("{:.2f}".format(obs_cpu[0][i+3]))



                n_proprio=45
                history_len=15
                num_z_encoder = 32



                encoder_input = np.zeros([1, n_proprio*history_len], dtype=np.float32)
                encoder_output = np.zeros([1, num_z_encoder], dtype=np.float32) 

                policy_input = np.zeros([1, n_proprio+num_z_encoder], dtype=np.float32) 


                # encoder_input[0,0:n_proprio]=obs

                hist_obs.append(obs) #11,1,45
                hist_obs.popleft() #10,1,45
                for i in range(history_len):#缓存历史观测
                    # encoder_input[0, i * n_proprio :   (i + 1) * n_proprio] = hist_obs[i]
                    encoder_input[0, i * n_proprio :   (i + 1) * n_proprio] = hist_obs[i]


                encoder_output_name = encoder.get_outputs()[0].name
                encoder_input_name = encoder.get_inputs()[0].name
                # for i in range(num_observations):
                #     encoder_input[0, i] = 0
                encoder_output = encoder.run([encoder_output_name], {encoder_input_name: encoder_input})[0]
                # print("encoder",encoder_output)
                teacher_latent = normalize_l2_numpy(encoder_output, axis=-1)
                
                # print("encoder_output:",encoder_output)
                for i in range(n_proprio):#缓存历史观测
                    policy_input[0, i] = obs[0][i]

                for i in range(num_z_encoder):
                    policy_input[0, i+(n_proprio)] = teacher_latent[0,i]
                # for i in range(history_len):#缓存历史观测
                #     hist_obs_input[0, i * n_proprio : (i + 1) * n_proprio] = hist_obs[i][0, :]
               
                policy_output_name = policy.get_outputs()[0].name
                policy_input_name = policy.get_inputs()[0].name

                action[:] = policy.run([policy_output_name], {policy_input_name: policy_input})[0]
                # print("encoder_output:",encoder_output)


                action = np.clip(action, -10,10)

                last_actions=action

                # action_flt=_low_pass_action_filter(action,last_actions)
                # last_actions=action
                action[0] *= 0.5
                action[3] *= 0.5 
                action[6] *= 0.5
                action[9] *= 0.5

                action_flt = action *0.25
                # 直接选择特定索引
                # action_flt[[0, 3, 6, 9]] *= Sim2simCfg.control.hip_scale_reduction

                joint_pos_target = action_flt + default_dof_pos
                target_q=joint_pos_target


                target_dq = np.zeros((12), dtype=np.double)
                # Generate PD control
                tau = pd_control(target_q, q, Sim2simCfg.robot_config.kps,
                                target_dq, dq, Sim2simCfg.robot_config.kds)  # Calc torques
                tau = np.clip(tau, -Sim2simCfg.robot_config.tau_limit, Sim2simCfg.robot_config.tau_limit)  # Clamp torques
                # torques = np.clip(tau, -torque_limits, torque_limits)
                data.ctrl = tau
            time.sleep(0.003)
            mujoco.mj_step(model, data)
            
        else:#air mode test
            obs = np.zeros([1, Sim2simCfg.env.n_proprio], dtype=np.float32) #1,45

            target_q = default_dof_pos
            # target_q[0]=0
            # target_q[1]=3
            # target_q[2]=3
            # target_q[3]=0
            # target_q[4]=3
            # target_q[5]=3     
            #print(eu_ang*57.3)
            target_dq = np.zeros((12), dtype=np.double)
            # Generate PD control
            tau = pd_control(target_q, q, Sim2simCfg.robot_config.kps,
                            target_dq, dq, Sim2simCfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -Sim2simCfg.robot_config.tau_limit, Sim2simCfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau

            mujoco.mj_step(model, data)
        if not np.array_equal(command, prev_command):
            print(
                "Use arrow keys to move the robot.\n",
                "'l': turn left\n",
                "'r': turn left\n",
                "UP: move forward,\n",
                "DOWN: move backward,\n",
                "LEFT: move left,\n" "RIGHT: move right.\n",
            )
            print(
                "x_velocity: {:.2f},\ny_velocity: {:.2f},\nangular_velocity: {:.2f}\n".format(
                    command[0], command[1], command[2]
                )
            )
            prev_command = command.copy()
        viewer.render()
        count_lowlevel += 1

    viewer.close()

    