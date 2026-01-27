import os
import sys
import numpy as np

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

    print(
        f"mean_variance on data shape: {data.shape}"
    )

    Mean = data.mean(axis=0)
    Std = data.std(axis=0)

    Std[0:1] = \
        Std[0:1].mean() / 1.0  # 根节点旋转速度
    Std[1:3] = \
        Std[1:3].mean() / 1.0  # 根节点 XZ 线速度
    Std[3:4] = \
        Std[3:4].mean() / 1.0  # 根节点 Y 高度
    Std[4: 4 + (joints_num - 1) * 3] = \
        Std[4: 4 + (joints_num - 1) * 3].mean() / 1.0  # 关节位置
    Std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = \
        Std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9].mean() / 1.0  # 关节旋转 (6D)
    Std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = \
        Std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3].mean() / 1.0  # 关节速度
    Std[4 + (joints_num - 1) * 9 + joints_num * 3:] = \
        Std[4 + (joints_num - 1) * 9 + joints_num * 3:].mean() / 1.0  # 脚步接触

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std


def main():
    # The given data is used to double-check if you are on the right track.
    reference_mean = np.load('./HumanML3D/Mean.npy')
    reference_std = np.load('./HumanML3D/Std.npy')

    data_dir = './HumanML3D/new_joint_vecs/'
    save_dir = './HumanML3D/'
    mean, std = mean_variance(data_dir, save_dir, joints_num=22)

    print(
        f"Compare data output mean and std result: {abs(mean - reference_mean).sum()}, {abs(std - reference_std).sum()}\n"
        f"If you see this line, you are on the right track!"
    )


if __name__ == "__main__":
    main()
