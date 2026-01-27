import os
import time
import torch

import codecs as cs
import pandas as pd
import numpy as np

from os.path import join as pjoin
from tqdm import tqdm

from human_body_prior.tools.omni_tools import copy2cpu

# Choose the device to run the body model on.
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from human_body_prior.body_model.body_model import BodyModel


def amass_to_pose(src_path, save_path):
    """
    将 AMASS 数据集中的 .npz 文件转换为关节位置 .npy 文件

    :param src_path: 输入的 AMASS .npz 文件路径
    :param save_path: 输出的关节位置 .npy 文件路径
    :return: 原始帧率 fps
    """

    """
    Y-Up: 
    - 这意味着世界的“向上”方向，也就是通常代表高度或海拔的方向，是沿着Y轴的正方向。
    - X轴通常指向“右”，Z轴指向“前”或“屏幕里”。Blender 和 3ds Max 默认使用这种坐标系。
    - 左手系!!!

    Z-Up: 
    - 这意味着世界的“向上”方向是沿着Z轴的正方向。
    - X轴指向“右”，Y轴指向“前”或“屏幕里”。Unity, Maya (默认)，以及许多CAD软件使用这种坐标系。
    - 右手系!!!

    需要转换的数据类型：
    - 转换不仅仅是简单的顶点位置，还可能包括：
        - 顶点位置 (Vertex Positions)： 最直接的转换。
        - 旋转 (Rotations)： 通常用欧拉角(Euler Angles)、四元数(Quaternions)或旋转矩阵(Rotation Matrices)表示。这是转换中最容易出错的部分。
        - 法线 (Normals) 和 切线 (Tangents)： 描述表面方向的向量。
        - 动画数据： 关键帧中的位置和旋转数据。
    - 核心思想： 转换的本质是重新映射坐标轴。我们需要找到一个变换，将旧坐标系的轴映射到新坐标系的轴上。

    顶点位置 (Point / Position) 的转换
    - 这是最简单的部分。我们只需要“交换”Y轴和Z轴的值，并注意符号以确保方向正确。

    假设在原始Y-Up坐标系中有一个点 P_old = (x, y, z)。
    - 在Z-Up坐标系中，我们希望：
        - 新的X轴 = 旧的X轴 -> x_new = x_old
        - 新的Y轴 = 旧的Z轴 -> y_new = z_old
        - 新的Z轴 = 旧的Y轴 -> z_new = y_old
    - 但是，这里有一个巨大的陷阱！直接交换会导致手性（Handedness）问题。
        - 直接交换 (x, y, z) -> (x, z, y) 实际上会执行一个绕X轴旋转-90度的变换，这会改变坐标系的手性（例如，从右手坐标系变成左手坐标系）。
        - 为了保持手性一致（例如，始终保持为右手坐标系），我们需要在交换后对其中一个轴取反。常见的做法是对新的Y轴（即旧的Z轴）取反：
    - 标准转换公式：
        - P_new = (x_old, -z_old, y_old)
    """
    # 坐标系转换矩阵 (Y-up -> Z-up)
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0]])  # Jason 2025-09-07: 这里没有对手性进行调整!!! 为什么？？？
    tgt_fps = 20

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
        body_model_object = male_body_model
    else:
        body_model_object = female_body_model

    """
    Jason 2025-10-17:
    原始的 AMASS 数据集帧率是 120 fps
    HUMANML3D 进行降采样处理，然后与 index.csv 中的起止帧对应，
    得到最终的 20 * N fps 数据 (N 根据每个数据集不同而不同)
    """
    # 降采样处理 (120 fps -> 20 fps)
    import scipy.signal

    num_frames = int(len(bdata['poses']) * tgt_fps / fps)
    bdata_poses = scipy.signal.resample(bdata['poses'], num_frames)
    bdata_trans = scipy.signal.resample(bdata['trans'], num_frames)

    # 构建模型输入参数
    body_parms = {
        'root_orient': torch.Tensor(bdata_poses[:, :3]).to(comp_device),
        'pose_body': torch.Tensor(bdata_poses[:, 3:66]).to(comp_device),
        'pose_hand': torch.Tensor(bdata_poses[:, 66:]).to(comp_device),
        'trans': torch.Tensor(bdata_trans).to(comp_device),
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=len(bdata_trans), axis=0)).to(comp_device),
    }

    # 通过人体模型计算关节位置
    # 包含 “full_pose” 字段：存储 root_orient, pose_body, pose_hand
    with torch.no_grad():
        body = body_model_object(**body_parms)

    # 坐标系转换 (Y-up -> Z-up)
    pose_seq_np = body.Jtr.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)

    # 保存处理后的关节数据
    np.save(save_path, pose_seq_np_n)

    return fps


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


def load_with_fallback(source_path):
    """
    尝试加载 source_path
    若不存在，则在其数据集根目录下递归查找同名文件
    """
    if os.path.exists(source_path):
        return np.load(source_path)

    filename = os.path.basename(source_path)

    # 推断数据集根目录
    # 例：./pose_data/KIT/3/kick_xxx.npy -> ./pose_data/KIT
    # 例：./pose_data/Eyes_Japan_Dataset/hamada/pose-06-hangon-hamada_poses.npy -> ./pose_data/EyesJapanDataset
    parts = source_path.split(os.sep)

    parts_0 = parts[0]  # ./pose_data
    parts_1 = parts[1]  # KIT or Eyes_Japan_Dataset

    if os.path.exists(os.path.join(parts_0, parts_1)):
        dataset_root = os.path.join(parts_0, parts_1)
    else:
        dataset_root = parts_0

    for root, _, files in os.walk(dataset_root):
        if filename in files:
            fallback_path = os.path.join(root, filename)
            print(f"[Fallback] {source_path} -> {fallback_path}")
            return np.load(fallback_path)

    raise FileNotFoundError(
        f"File not found: {source_path} (also not found under {dataset_root})"
    )


def main():
    """
    Jason 2025-08-14:
    SMPL（Skinned Multi-Person Linear Model）和DMPL（Dynamic SMPL）是三维人体建模中紧密关联但功能侧重点不同的模型，
    它们在计算机图形学、动画、虚拟现实等领域有重要应用。

    SMPL:
    - SMPL是一种参数化人体模型，通过解耦形状（shape）和姿态（pose）参数生成三维人体网格（含6890个顶点）。
    - SMPL（Skinned Multi-Person Linear Model）：静态人体模型。它负责生成一个“静止”的、具有正确肌肉线条和皮肤蒙皮的人体三维网格。
    - 输入：它的参数主要控制形状（Shape） 和姿态（Pose）。
        - β (Shape参数): 一个10维或300维的向量，控制人的高矮胖瘦等体型特征。
        - θ (Pose参数): 一个72维的向量（24个关节，每个关节3个轴的角度），控制人体的全局旋转和24个关节的旋转。
    - 输出：
        - 一个包含6890个顶点的三维人体网格。

    DMPL:
    - DMPL（Dynamic Multi-Person Linear Model）：动态软组织模型。它本身不生成完整的人体网格，而是为SMPL模型增加软组织在运动过程中的动态变形效果，如肌肉的收缩、膨胀和抖动。
    - 它的参数主要控制软组织运动（Soft Tissue Movement）。
    - 输入：DMPL是SMPL的动态扩展版本，在保留基础参数（β, θ）的基础上，增加了对软组织动力学（如肌肉颤动、脂肪抖动）的模拟能力。
        - β (Shape参数): 与SMPL共享相同的形状参数。
        - θ (Pose参数): 与SMPL共享相同的姿态参数。
        - ψ (DMPL参数): 一个4维或8维的向量，可以理解为控制“肌肉兴奋度”的参数，用于驱动软组织的动态效果。【新增】
    - 输出：
        - 不是完整的网格，而是一组顶点偏移量（Vertex Offsets）。
        - 这些偏移量描述了由于软组织运动（如肌肉收缩、脂肪抖动）每个顶点应该移动的方向和距离。

    它们的配合使用可以概括为：以SMPL为基础，用DMPL的参数为其增添动态细节。
    SMPL和DMPL配合使用的最终目标是生成一个既包含正确姿态和形状，又包含逼真软组织动态的顶点位置 V_final。
    其数学表达可以简化为：
    - V_final = SMPL(β, θ) + DMPL(β, θ, ψ)

    步骤分解：
    1. 生成基础网格：
        - 首先，将形状参数 β 和姿态参数 θ 输入到标准的SMPL模型中，计算得到基础的三维网格顶点位置 V_smpl。
        - V_smpl = SMPL(β, θ)
    2. 计算动态偏移：
        - 将相同的 β 和 θ，再加上DMPL专属的参数 ψ，一起输入到DMPL模型中。DMPL模型会输出一个与 V_smpl 顶点数量相同（6890个）的偏移量向量 ΔV_dmpl。
        - ΔV_dmpl = DMPL(β, θ, ψ)
    3. 合成最终网格：
        - 将SMPL生成的基础网格顶点与DMPL计算出的动态偏移量相加，得到最终的、具有动态细节的顶点位置。
        - V_final = V_smpl + ΔV_dmpl
    4. 蒙皮与渲染：
        - 使用SMPL原有的蒙皮权重函数，对最终的顶点 V_final 进行蒙皮，并将其渲染成图像或用于其他应用。
        注意，蒙皮权重是在 V_smpl 上定义的，但直接应用于 V_final 也能得到合理的效果，因为 ΔV_dmpl 是相对偏移。

    这种SMPL+DMPL的模型组合（有时被统称为SMPL-D）
    """

    # number of body parameters
    # 形状参数（β）：控制身高、体型（胖瘦）等静态特征，通过PCA降维实现高效参数化
    num_betas = 10

    # USE: SMPL-H + DMPL model
    male_body_model_smplh_path = 'body_model/smplh/male/model.npz'
    male_body_model_dmpl_path = 'body_model/dmpls/male/model.npz'

    female_body_model_smplh_path = 'body_model/smplh/female/model.npz'
    female_body_model_dmpl_path = 'body_model/dmpls/female/model.npz'

    # number of DMPL parameters
    # DMPL参数（ψ）：控制肌肉动态、脂肪抖动等软组织运动，通常维度较低（4或8维）
    num_dmpls = 8

    male_body_model = BodyModel(smpl_file_path=male_body_model_smplh_path,
                                dmpl_file_path=male_body_model_dmpl_path,
                                num_betas=num_betas,
                                num_dmpls=num_dmpls, ).to(comp_device)
    female_body_model = BodyModel(smpl_file_path=female_body_model_smplh_path,
                                  dmpl_file_path=female_body_model_dmpl_path,
                                  num_betas=num_betas,
                                  num_dmpls=num_dmpls, ).to(comp_device)

    # USE: SMPL-X model
    # male_body_model_smplx_path = 'body_model/smplx/SMPLX_MALE.npz'
    # female_body_model_smplx_path = 'body_model/smplx/SMPLX_FEMALE.npz'
    #
    # male_body_model = BodyModel(smpl_file_path=male_body_model_smplx_path,
    #                             num_betas=num_betas).to(comp_device)
    # female_body_model = BodyModel(smpl_file_path=female_body_model_smplx_path,
    #                               num_betas=num_betas).to(comp_device)

    faces = copy2cpu(male_body_model.f)  # Jason 2025-09-06: 面部信息，好像没用上

    # 递归扫描 AMASS 数据集目录
    # 收集所有数据文件的路径
    paths = []
    folders = []
    dataset_names = []

    VALID_EXTENSIONS = (".npz", ".npy")  # AMASS 实际只需要 .npz

    for root, dirs, files in os.walk('./motion_data/amass_data'):
        folders.append(root)

        # 防止路径层级变化导致越界
        parts = root.replace("\\", "/").split("/")
        if len(parts) >= 3:
            dataset_name = parts[2]
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)

        for name in files:
            # 关键过滤条件
            if not name.lower().endswith(VALID_EXTENSIONS):
                continue

            paths.append(os.path.join(root, name))

    print(
        f"Found {len(paths)} files in {len(folders)} folders from {len(dataset_names)} datasets."
    )

    # 在输出目录中复制AMASS的原始目录结构
    # 使用 exist_ok=True 避免重复创建
    save_root = './pose_data'
    save_folders = [folder.replace('./motion_data/amass_data', './pose_data') for folder in folders]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)
    group_path = [[path for path in paths if name in path] for name in dataset_names]

    print(
        f"Created {len(save_folders)} folders in {save_root}."
    )

    # --------------------------------------------------

    group_path = group_path
    all_count = sum([len(paths) for paths in group_path])
    current_count = 0

    # --------------------------------------------------

    # 处理每个数据集
    # 将 AMASS 数据 .npz 转换为关节位置 .npy (关节位置点的坐标)
    for paths in group_path:
        dataset_name = paths[0].split('/')[2]
        pbar = tqdm(paths)
        pbar.set_description('Processing: %s' % dataset_name)
        fps = 0

        for path in pbar:
            save_path = path.replace('./motion_data/amass_data', './pose_data')
            save_path = save_path[:-3] + 'npy'
            fps = amass_to_pose(path, save_path)

        current_count += len(paths)
        print('Processed / All (fps %d): %d/%d' % (fps, current_count, all_count))
        time.sleep(0.5)

    # --------------------------------------------------

    index_path = './index.csv'
    save_dir = './joints'
    index_file = pd.read_csv(index_path)
    total_amount = index_file.shape[0]
    fps = 20  # 基础帧率

    # 处理索引文件中的每个动作片段
    for i in tqdm(range(total_amount)):
        # 从index.csv获取元数据
        source_path = index_file.loc[i]['source_path']
        new_name = index_file.loc[i]['new_name']

        # 加载之前生成的关节数据
        data = load_with_fallback(source_path)
        start_frame = index_file.loc[i]['start_frame']
        end_frame = index_file.loc[i]['end_frame']

        # FIXME: 加入截断数据操作

        # 数据集特定处理：开头裁剪若干帧（相当于延时几秒开始记录）
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

        # 创建镜像版本并保存
        data_m = swap_left_right(data)
        np.save(pjoin(save_dir, new_name), data)  # 原始
        np.save(pjoin(save_dir, 'M' + new_name), data_m)  # 镜像


if __name__ == "__main__":
    main()
