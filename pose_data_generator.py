import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from human_body_prior.tools.omni_tools import copy2cpu as c2c

flag_run_raw_pose_processing = False
flag_run_motion_representation = True
flag_run_calculate_mean_variance = False
flag_run_animation = False

# ====================================================================================================

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# ====================================================================================================

if not flag_run_raw_pose_processing:
    pass

else:

    # Choose the device to run the body model on.
    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from human_body_prior.body_model.body_model import BodyModel

    """
    Jason 2025-08-14:
    SMPL（Skinned Multi-Person Linear Model）和DMPL（Dynamic SMPL）是三维人体建模中紧密关联但功能侧重点不同的模型，
    它们在计算机图形学、动画、虚拟现实等领域有重要应用。
    
    - SMPL是一种参数化人体模型，通过解耦形状（shape）和姿态（pose）参数生成三维人体网格（含6890个顶点）。
    - DMPL是SMPL的动态扩展版本，在保留基础参数（β, θ）的基础上，增加了对软组织动力学（如肌肉颤动、脂肪抖动）的模拟能力。
    """
    male_bm_path = 'body_model/smplh/male/model.npz'
    male_dmpl_path = 'body_model/dmpls/male/model.npz'

    female_bm_path = 'body_model/smplh/female/model.npz'
    female_dmpl_path = 'body_model/dmpls/female/model.npz'

    num_betas = 10  # number of body parameters 形状参数（β）：控制身高、体型（胖瘦）等静态特征，通过PCA降维实现高效参数化
    num_dmpls = 8  # number of DMPL parameters

    male_bm = BodyModel(bm_fname=male_bm_path,
                        num_betas=num_betas,
                        num_dmpls=num_dmpls,
                        dmpl_fname=male_dmpl_path).to(comp_device)

    faces = c2c(male_bm.f)

    female_bm = BodyModel(bm_fname=female_bm_path,
                          num_betas=num_betas,
                          num_dmpls=num_dmpls,
                          dmpl_fname=female_dmpl_path).to(comp_device)

    # 递归扫描 AMASS 数据集目录
    # 收集所有数据文件的路径
    paths = []
    folders = []
    dataset_names = []

    for root, dirs, files in os.walk('./motion_data/amass_data'):
        #     print(root, dirs, files)
        #     for folder in dirs:
        #         folders.append(os.path.join(root, folder))
        folders.append(root)
        for name in files:
            dataset_name = root.split('/')[2]
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)
            paths.append(os.path.join(root, name))

    # 在输出目录中复制AMASS的原始目录结构
    # 使用 exist_ok=True 避免重复创建
    save_root = './pose_data'
    save_folders = [folder.replace('./motion_data/amass_data', './pose_data') for folder in folders]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)
    group_path = [[path for path in paths if name in path] for name in dataset_names]

    trans_matrix = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0]])
    ex_fps = 20


    def amass_to_pose(src_path, save_path):
        # 加载 AMASS 的 npz 数据
        bdata = np.load(src_path, allow_pickle=True)
        fps = 0

        try:
            fps = bdata['mocap_framerate']
            frame_number = bdata['trans'].shape[0]
        except:
            #         print(list(bdata.keys()))
            return fps

        frame_id = 0  # frame id of the mocap sequence
        pose_seq = []
        if bdata['gender'] == 'male':
            bm = male_bm
        else:
            bm = female_bm

        # 降采样处理 (120fps -> 20fps)
        down_sample = int(fps / ex_fps)
        #     print(frame_number)
        #     print(fps)

        bdata_poses = bdata['poses'][::down_sample, ...]
        bdata_trans = bdata['trans'][::down_sample, ...]

        # 构建模型输入参数
        body_parms = {
            'root_orient': torch.Tensor(bdata_poses[:, :3]).to(comp_device),
            'pose_body': torch.Tensor(bdata_poses[:, 3:66]).to(comp_device),
            'pose_hand': torch.Tensor(bdata_poses[:, 66:]).to(comp_device),
            'trans': torch.Tensor(bdata_trans).to(comp_device),
            'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=len(bdata_trans), axis=0)).to(comp_device),
        }

        # 通过人体模型计算关节位置
        with torch.no_grad():
            body = bm(**body_parms)

        # 坐标系转换 (Y-up -> Z-up)
        pose_seq_np = body.Jtr.detach().cpu().numpy()
        pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)

        # 保存处理后的关节数据
        np.save(save_path, pose_seq_np_n)

        return fps


    group_path = group_path
    all_count = sum([len(paths) for paths in group_path])
    cur_count = 0

    # --------------------------------------------------

    import time

    for paths in group_path:
        dataset_name = paths[0].split('/')[2]
        pbar = tqdm(paths)
        pbar.set_description('Processing: %s' % dataset_name)
        fps = 0
        for path in pbar:
            save_path = path.replace('./motion_data/amass_data', './pose_data')
            save_path = save_path[:-3] + 'npy'
            fps = amass_to_pose(path, save_path)

        cur_count += len(paths)
        print('Processed / All (fps %d): %d/%d' % (fps, cur_count, all_count))
        time.sleep(0.5)

    # --------------------------------------------------

    import codecs as cs
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from os.path import join as pjoin


    # 左右镜像翻转函数
    def swap_left_right(data):
        assert len(data.shape) == 3 and data.shape[-1] == 3
        data = data.copy()

        # 翻转X轴
        data[..., 0] *= -1

        right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
        left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
        left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
        right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]

        # 交换左右身体部位索引
        tmp = data[:, right_chain]
        data[:, right_chain] = data[:, left_chain]
        data[:, left_chain] = tmp

        if data.shape[1] > 24:
            tmp = data[:, right_hand_chain]
            data[:, right_hand_chain] = data[:, left_hand_chain]
            data[:, left_hand_chain] = tmp
        return data


    index_path = './index.csv'
    save_dir = './joints'
    index_file = pd.read_csv(index_path)
    total_amount = index_file.shape[0]
    fps = 20

    # 处理索引文件中的每个动作片段
    for i in tqdm(range(total_amount)):
        # 从index.csv获取元数据
        source_path = index_file.loc[i]['source_path']
        new_name = index_file.loc[i]['new_name']

        # 加载之前生成的关节数据
        data = np.load(source_path)
        start_frame = index_file.loc[i]['start_frame']
        end_frame = index_file.loc[i]['end_frame']

        # 数据集特定处理
        if 'humanact12' not in source_path:
            if 'Eyes_Japan_Dataset' in source_path:
                data = data[3 * fps:]
            if 'MPI_HDM05' in source_path:
                data = data[3 * fps:]
            if 'TotalCapture' in source_path:
                data = data[1 * fps:]
            if 'MPI_Limits' in source_path:
                data = data[1 * fps:]
            if 'Transitions_mocap' in source_path:
                data = data[int(0.5 * fps):]
            data = data[start_frame:end_frame]
            data[..., 0] *= -1

        # 创建镜像版本并保存
        data_m = swap_left_right(data)
        np.save(pjoin(save_dir, new_name), data)  # 原始
        np.save(pjoin(save_dir, 'M' + new_name), data_m)  # 镜像

# ====================================================================================================

if not flag_run_motion_representation:
    pass

else:

    from os.path import join as pjoin

    from common.skeleton import Skeleton
    import numpy as np
    import os
    from common.quaternion import *
    from paramUtil import *

    import torch
    from tqdm import tqdm
    import os


    def uniform_skeleton(positions, target_offset):
        src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
        src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
        src_offset = src_offset.numpy()
        tgt_offset = target_offset.numpy()

        # print(src_offset)
        # print(tgt_offset)

        '''Calculate Scale Ratio as the ratio of legs'''
        src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
        tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

        scale_rt = tgt_leg_len / src_leg_len
        # print(scale_rt)
        src_root_pos = positions[:, 0]
        tgt_root_pos = src_root_pos * scale_rt

        '''Inverse Kinematics'''
        quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
        # print(quat_params.shape)

        '''Forward Kinematics'''
        src_skel.set_offset(target_offset)
        new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
        return new_joints


    def process_file(positions, feet_thre):
        # (seq_len, joints_num, 3)
        #     '''Down Sample'''
        #     positions = positions[::ds_num]

        '''Uniform Skeleton'''
        positions = uniform_skeleton(positions, tgt_offsets)

        '''Put on Floor'''
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height
        #     print(floor_height)

        #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

        '''XZ at origin'''
        root_pos_init = positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz

        # '''Move the first pose to origin '''
        # root_pos_init = positions[0]
        # positions = positions - root_pos_init[0]

        '''All initially face Z+'''
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

        #     print(forward_init)

        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

        positions_b = positions.copy()

        positions = qrot_np(root_quat_init, positions)

        #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

        '''New ground truth positions'''
        global_positions = positions.copy()

        # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
        # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
        # plt.xlabel('x')
        # plt.ylabel('z')
        # plt.axis('equal')
        # plt.show()

        """ Get Foot Contacts """

        def foot_detect(positions, thres):
            velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            #     feet_l_h = positions[:-1,fid_l,1]
            #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
            feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            #     feet_r_h = positions[:-1,fid_r,1]
            #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
            return feet_l, feet_r

        #
        feet_l, feet_r = foot_detect(positions, feet_thre)
        # feet_l, feet_r = foot_detect(positions, 0.002)

        '''Quaternion and Cartesian representation'''
        r_rot = None

        def get_rifke(positions):
            '''Local pose'''
            positions[..., 0] -= positions[:, 0:1, 0]
            positions[..., 2] -= positions[:, 0:1, 2]
            '''All pose face Z+'''
            positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
            return positions

        def get_quaternion(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            # (seq_len, joints_num, 4)
            quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

            '''Fix Quaternion Discontinuity'''
            quat_params = qfix(quat_params)
            # (seq_len, 4)
            r_rot = quat_params[:, 0].copy()
            #     print(r_rot[0])
            '''Root Linear Velocity'''
            # (seq_len - 1, 3)
            velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
            #     print(r_rot.shape, velocity.shape)
            velocity = qrot_np(r_rot[1:], velocity)
            '''Root Angular Velocity'''
            # (seq_len - 1, 4)
            r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
            quat_params[1:, 0] = r_velocity
            # (seq_len, joints_num, 4)
            return quat_params, r_velocity, velocity, r_rot

        def get_cont6d_params(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            # (seq_len, joints_num, 4)
            quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

            '''Quaternion to continuous 6D'''
            cont_6d_params = quaternion_to_cont6d_np(quat_params)
            # (seq_len, 4)
            r_rot = quat_params[:, 0].copy()
            #     print(r_rot[0])
            '''Root Linear Velocity'''
            # (seq_len - 1, 3)
            velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
            #     print(r_rot.shape, velocity.shape)
            velocity = qrot_np(r_rot[1:], velocity)
            '''Root Angular Velocity'''
            # (seq_len - 1, 4)
            r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
            # (seq_len, joints_num, 4)
            return cont_6d_params, r_velocity, velocity, r_rot

        cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
        positions = get_rifke(positions)

        #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
        #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

        # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
        # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
        # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
        # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
        # plt.xlabel('x')
        # plt.ylabel('z')
        # plt.axis('equal')
        # plt.show()

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
        #     print(data.shape, local_vel.shape)
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
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get Y-axis rotation from rotation velocity'''
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        '''Add Y-axis rotation to root position'''
        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=-2)

        r_pos[..., 1] = data[..., 3]
        return r_rot_quat, r_pos


    def recover_from_rot(data, joints_num, skeleton):
        r_rot_quat, r_pos = recover_root_rot_pos(data)

        r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

        start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
        end_indx = start_indx + (joints_num - 1) * 6
        cont6d_params = data[..., start_indx:end_indx]
        #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
        cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
        cont6d_params = cont6d_params.view(-1, joints_num, 6)

        positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

        return positions


    def recover_from_ric(data, joints_num):
        r_rot_quat, r_pos = recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        return positions


    # The given data is used to double check if you are on the right track.
    reference1 = np.load('./HumanML3D/new_joints/012314.npy')
    reference2 = np.load('./HumanML3D/new_joint_vecs/012314.npy')

    '''
    For HumanML3D Dataset
    '''

    if __name__ == "__main__":
        example_id = "000021"
        # Lower legs
        l_idx1, l_idx2 = 5, 8
        # Right/Left foot
        fid_r, fid_l = [8, 11], [7, 10]
        # Face direction, r_hip, l_hip, sdr_r, sdr_l
        face_joint_indx = [2, 1, 17, 16]
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
        tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
        # (joints_num, 3)
        tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
        # print(tgt_offsets)

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
        #         print(source_file)
        #         break

        print('Total clips: %d, Frames: %d, Duration: %fm' %
              (len(source_list), frame_num, frame_num / 20 / 60))

    reference1_1 = np.load('./HumanML3D/new_joints/012314.npy')
    reference2_1 = np.load('./HumanML3D/new_joint_vecs/012314.npy')

    abs(reference1 - reference1_1).sum()
    abs(reference2 - reference2_1).sum()

# ====================================================================================================

if not flag_run_calculate_mean_variance:
    pass

else:

    import numpy as np
    import sys
    import os
    from os.path import join as pjoin


    # root_rot_velocity (B, seq_len, 1)
    # root_linear_velocity (B, seq_len, 2)
    # root_y (B, seq_len, 1)
    # ric_data (B, seq_len, (joint_num - 1)*3)
    # rot_data (B, seq_len, (joint_num - 1)*6)
    # local_velocity (B, seq_len, joint_num*3)
    # foot contact (B, seq_len, 4)
    def mean_variance(data_dir, save_dir, joints_num):
        file_list = os.listdir(data_dir)
        data_list = []

        for file in file_list:
            data = np.load(pjoin(data_dir, file))
            if np.isnan(data).any():
                print(file)
                continue
            data_list.append(data)

        data = np.concatenate(data_list, axis=0)
        print(data.shape)
        Mean = data.mean(axis=0)
        Std = data.std(axis=0)
        Std[0:1] = Std[0:1].mean() / 1.0
        Std[1:3] = Std[1:3].mean() / 1.0
        Std[3:4] = Std[3:4].mean() / 1.0
        Std[4: 4 + (joints_num - 1) * 3] = Std[4: 4 + (joints_num - 1) * 3].mean() / 1.0
        Std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = Std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9].mean() / 1.0
        Std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = Std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3].mean() / 1.0
        Std[4 + (joints_num - 1) * 9 + joints_num * 3:] = Std[4 + (joints_num - 1) * 9 + joints_num * 3:].mean() / 1.0

        assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

        np.save(pjoin(save_dir, 'Mean.npy'), Mean)
        np.save(pjoin(save_dir, 'Std.npy'), Std)

        return Mean, Std


    # The given data is used to double check if you are on the right track.
    reference1 = np.load('./HumanML3D/Mean.npy')
    reference2 = np.load('./HumanML3D/Std.npy')

    if __name__ == '__main__':
        data_dir = './HumanML3D/new_joint_vecs/'
        save_dir = './HumanML3D/'
        mean, std = mean_variance(data_dir, save_dir, 22)
    #     print(mean)
    #     print(Std)

    abs(mean - reference1).sum()
    abs(std - reference2).sum()

# ====================================================================================================

if not flag_run_animation:
    pass

else:

    import os
    from os.path import join as pjoin
    from tqdm import tqdm
    import numpy as np
    # %%
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import mpl_toolkits.mplot3d.axes3d as p3


    def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
        #     matplotlib.use('Agg')

        title_sp = title.split(' ')
        if len(title_sp) > 10:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

        def init():
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([0, radius])
            # print(title)
            fig.suptitle(title, fontsize=20)
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        #         return ax

        # (seq_len, joints_num, 3)
        data = joints.copy().reshape(len(joints), -1, 3)
        fig = plt.figure(figsize=figsize)
        ax = p3.Axes3D(fig)
        init()
        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)
        colors = ['red', 'blue', 'black', 'red', 'blue',
                  'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
                  'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
        frame_number = data.shape[0]
        #     print(data.shape)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]

        #     print(trajec.shape)

        def update(index):
            #         print(index)
            ax.lines = []
            ax.collections = []
            ax.view_init(elev=120, azim=-90)
            ax.dist = 7.5
            #         ax =
            plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])
            #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

            if index > 1:
                ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                          color='blue')
            #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

            for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
                #             print(color)
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
            #         print(trajec[:index, 0].shape)

            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

        ani.save(save_path, fps=fps)
        plt.close()


    # %%
    src_dir = './HumanML3D/new_joints/'
    tgt_ani_dir = "./HumanML3D/animations/"
    # %%
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    os.makedirs(tgt_ani_dir, exist_ok=True)
    # %%
    npy_files = os.listdir(src_dir)
    npy_files = sorted(npy_files)
    # npy_files = npy_files[:10]

    for npy_file in tqdm(npy_files):
        data = np.load(pjoin(src_dir, npy_file))
        save_path = pjoin(tgt_ani_dir, npy_file[:-3] + 'mp4')
        if os.path.exists(save_path):
            continue
        #   You may set the title on your own.
        plot_3d_motion(save_path, kinematic_chain, data, title="None", fps=20, radius=4)

# ====================================================================================================
