import os
import numpy as np
import torch

from os.path import join as pjoin
from tqdm import tqdm

from common.skeleton import Skeleton
from common.quaternion import *
from paramUtil import *

# global variables
leg_idx1, leg_idx2 = None, None
foot_right_idx, foot_left_idx = None, None
face_joint_idx = None
n_raw_offsets = None
kinematic_chain = None
tgt_offsets = None


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[leg_idx1]).max() + np.abs(src_offset[leg_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[leg_idx1]).max() + np.abs(tgt_offset[leg_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_idx)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)

    return new_joints


def foot_detect(positions, thres):
    velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, foot_left_idx, 0] - positions[:-1, foot_left_idx, 0]) ** 2
    feet_l_y = (positions[1:, foot_left_idx, 1] - positions[:-1, foot_left_idx, 1]) ** 2
    feet_l_z = (positions[1:, foot_left_idx, 2] - positions[:-1, foot_left_idx, 2]) ** 2
    #     feet_l_h = positions[:-1,foot_left_idx,1]
    #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, foot_right_idx, 0] - positions[:-1, foot_right_idx, 0]) ** 2
    feet_r_y = (positions[1:, foot_right_idx, 1] - positions[:-1, foot_right_idx, 1]) ** 2
    feet_r_z = (positions[1:, foot_right_idx, 2] - positions[:-1, foot_right_idx, 2]) ** 2
    #     feet_r_h = positions[:-1,foot_right_idx,1]
    #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
    return feet_l, feet_r


def get_rifke(positions, r_rot):
    '''Local pose'''
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]

    '''All pose face Z+'''
    positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)

    return positions


def get_quaternion(positions):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")  # (seq_len, joints_num, 4)
    quat_params = skel.inverse_kinematics_np(positions, face_joint_idx, smooth_forward=False)

    '''Fix Quaternion Discontinuity'''
    quat_params = qfix(quat_params)  # (seq_len, 4)
    root_rot = quat_params[:, 0].copy()

    '''Root Linear Velocity'''
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()  # (seq_len - 1, 3)
    velocity = qrot_np(root_rot[1:], velocity)

    '''Root Angular Velocity'''
    root_velocity = qmul_np(root_rot[1:], qinv_np(root_rot[:-1]))  # (seq_len - 1, 4)

    quat_params[1:, 0] = root_velocity  # (seq_len, joints_num, 4)

    return quat_params, root_velocity, velocity, root_rot


def get_cont6d_params(positions):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")  # (seq_len, joints_num, 4)

    quat_params = skel.inverse_kinematics_np(positions, face_joint_idx, smooth_forward=True)

    '''Quaternion to continuous 6D'''
    cont_6d_params = quaternion_to_cont6d_np(quat_params)  # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()

    '''Root Linear Velocity'''
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()  # (seq_len - 1, 3)
    velocity = qrot_np(r_rot[1:], velocity)

    '''Root Angular Velocity'''
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # (seq_len - 1, 4)
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot


def process_file(positions, feet_threshold):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # print(floor_height)
    # plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    hip_right, hip_left, shoulder_right, shoulder_left = face_joint_idx
    across1 = root_pos_init[hip_right] - root_pos_init[hip_left]
    across2 = root_pos_init[shoulder_right] - root_pos_init[shoulder_left]

    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]  # forward (3,)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()
    positions = qrot_np(root_quat_init, positions)

    # plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """
    feet_l, feet_r = foot_detect(positions, feet_threshold)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions, r_rot)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]

    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    root_rot_ang = torch.zeros_like(rot_vel).to(data.device)

    '''Get Y-axis rotation from rotation velocity'''
    root_rot_ang[..., 1:] = rot_vel[..., :-1]
    root_rot_ang = torch.cumsum(root_rot_ang, dim=-1)

    root_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    root_rot_quat[..., 0] = torch.cos(root_rot_ang)
    root_rot_quat[..., 2] = torch.sin(root_rot_ang)

    root_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    root_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    '''Add Y-axis rotation to root position'''
    root_pos = qrot(qinv(root_rot_quat), root_pos)

    root_pos = torch.cumsum(root_pos, dim=-2)

    root_pos[..., 1] = data[..., 3]

    return root_rot_quat, root_pos


def recover_from_rot(data, joints_num, skeleton):
    """
    Recover global joint positions from rotation data using continuous 6D rotation representation.

    Args:
        data (torch.Tensor): Input tensor containing rotation data with shape
                             (..., feature_dim). The feature_dim includes root
                             rotation velocity, root linear velocity, root height,
                             and joint rotation data.
        joints_num (int): The total number of joints in the skeleton.
        skeleton (Skeleton): The skeleton object used for forward kinematics.

    Returns:
        torch.Tensor: Tensor of recovered global joint positions with shape
                      (..., joints_num, 3). The positions include the root
                      joint and all other joints in the skeleton.
    """
    # Recover root rotation quaternion and root position from the input data
    root_rot_quat, root_pos = recover_root_rot_pos(data)

    # Convert root rotation quaternion to continuous 6D representation
    root_rot_cont6d = quaternion_to_cont6d(root_rot_quat)

    # Extract joint rotation data from the input tensor
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]

    # Concatenate root rotation with joint rotation data
    cont6d_params = torch.cat([root_rot_cont6d, cont6d_params], dim=-1)

    # Reshape the rotation data to match the skeleton's joint structure
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    # Perform forward kinematics to recover global joint positions
    positions = skeleton.forward_kinematics_cont6d(cont6d_params, root_pos)

    return positions


def recover_from_ric(data, joints_num):
    """
    Recover global joint positions from rotation-invariant coordinates (RIC) data.

    Args:
        data (torch.Tensor): Input tensor containing RIC data with shape
                             (..., feature_dim). The feature_dim includes root
                             rotation velocity, root linear velocity, root height,
                             and local joint positions.
        joints_num (int): The total number of joints in the skeleton.

    Returns:
        torch.Tensor: Tensor of recovered global joint positions with shape
                      (..., joints_num, 3). The positions include the root
                      joint and all other joints in the skeleton.
    """
    # Recover root rotation quaternion and root position from the input data
    root_rot_quat, root_pos = recover_root_rot_pos(data)

    # Extract local joint positions from the input data
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Apply inverse root rotation to the local joint positions
    positions = qrot(qinv(root_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # Add the root XZ position to the local joint positions
    positions[..., 0] += root_pos[..., 0:1]
    positions[..., 2] += root_pos[..., 2:3]

    # Concatenate the root position with the joint positions to form the full skeleton
    positions = torch.cat([root_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def main():
    global leg_idx1, leg_idx2
    global foot_right_idx, foot_left_idx
    global face_joint_idx
    global n_raw_offsets, kinematic_chain, tgt_offsets

    example_id = "000021"

    # Lower legs
    leg_idx1, leg_idx2 = 5, 8

    # Right/Left foot
    foot_right_idx, foot_left_idx = [8, 11], [7, 10]

    # Face direction, right_hip, left_hip, shoulder_right, shoulder_left
    face_joint_idx = [2, 1, 17, 16]

    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22

    # ds_num = 8
    data_dir = './joints/'
    save_dir1 = './HumanML3D/new_joints/'
    save_dir2 = './HumanML3D/new_joint_vecs/'

    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Get offsets of target skeleton
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)

    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')  # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    source_list = os.listdir(data_dir)

    frame_num = 0
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        try:
            data, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
            rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
            np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())
            np.save(pjoin(save_dir2, source_file), data)
            frame_num += data.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print(
        f"Total clips: {len(source_list)}, Frames: {frame_num}, Duration: {frame_num / 20 / 60}m"
    )

    # --------------------------------------------------

    # The given data is used to double check if you are on the right track.
    reference1 = np.load('./HumanML3D/new_joints/012314.npy')
    reference2 = np.load('./HumanML3D/new_joint_vecs/012314.npy')

    reference1_1 = np.load('./HumanML3D/new_joints/012314.npy')
    reference2_1 = np.load('./HumanML3D/new_joint_vecs/012314.npy')

    print(
        f"Compare data {abs(reference1 - reference1_1).sum()}, {abs(reference2 - reference2_1).sum()}\n"
        f"If you see this line, you are on the right track!"
    )

    # --------------------------------------------------

    """
    Jason 2026-01-27:
    | 属性             | `new_joints`                         | `new_joint_vecs`                     |
    | --------------- | ------------------------------------- | ----------------------------------- |
    | 数据类型         | 3D关节位置 `(seq_len, joints_num, 3)`  | 特征向量 `(seq_len, feature_dim)`     |
    | 是否包含速度信息  | 否                                    | 是（根节点速度 + 局部关节速度）          |
    | 是否旋转不变     | 否                                    | 是（RIFKE + cont6D）                  |
    | 是否包含脚步接触  | 否                                    | 是                                   |
    | 主要用途         | 可视化 / 渲染                          | 模型训练 / 动作表示                    |
    | 来源            | 对齐后的关节位置                        | 对齐 + IK + 旋转表示 + 速度 + 脚步接触   |
    """


if __name__ == "__main__":
    main()
