# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.12.13

import numpy as np

import torch
import torch.nn as nn

# from smplx.lbs import lbs
from human_body_prior.body_model.lbs import lbs
import sys


class BodyModel(nn.Module):

    def __init__(
            self,
            # SMPL parameters
            smpl_file_path,
            # DMPL parameters (optional)
            dmpl_file_path=None,
            num_betas=10,
            num_dmpls=None,
            num_expressions=80,
            use_posedirs=True,
            dtype=torch.float32,
            persistant_buffer=False
    ):
        '''
        初始化方法负责加载模型参数并设置初始状态：
        - 模型文件加载：从NPZ文件加载SMPL模型参数
        - 模型类型识别：根据关节数量自动识别模型类型（SMPL:69, SMPL-H:153, SMPL-X:162等）
        - 参数初始化：注册各种模型参数为PyTorch缓冲区（buffer）
        - 组件注册：使用comp_register方法将参数注册为模型的持久化缓冲区

        :param smpl_file_path: path to a SMPL model as pkl file
        :param num_betas: number of shape parameters to include.
        :param device: default on gpu
        :param dtype: float precision of the computations
        :return: verts, trans, pose, betas
        '''

        super(BodyModel, self).__init__()

        self.dtype = dtype

        # -- Load SMPL params --
        if '.npz' in smpl_file_path:
            smpl_dict = np.load(smpl_file_path, encoding='latin1')
        else:
            raise ValueError('smpl_file_path should be either a .pkl nor .npz file')

        # these are supposed for later convenient look up
        self.num_betas = num_betas
        self.num_dmpls = num_dmpls
        self.num_expressions = num_expressions

        njoints = smpl_dict['posedirs'].shape[2] // 3
        self.model_type = {69: 'smpl',
                           153: 'smplh',
                           162: 'smplx',
                           45: 'mano',
                           105: 'animal_horse',
                           102: 'animal_dog', }[njoints]

        assert self.model_type in ['smpl',
                                   'smplh',
                                   'smplx',
                                   'mano',
                                   'mano',
                                   'animal_horse',
                                   'animal_dog'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano.')

        self.use_dmpl = False
        if num_dmpls is not None:
            if dmpl_file_path is not None:
                self.use_dmpl = True
            else:
                raise (ValueError('dmpl_file_path should be provided when using dmpls!'))

        if self.use_dmpl and self.model_type in ['smplx', 'mano', 'animal_horse', 'animal_dog']:
            raise (NotImplementedError('DMPLs only work with SMPL/SMPLH models for now.'))

        # Mean template vertices
        self.comp_register('init_v_template', torch.tensor(smpl_dict['v_template'][None], dtype=dtype), persistent=persistant_buffer)

        self.comp_register('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32), persistent=persistant_buffer)

        num_total_betas = smpl_dict['shapedirs'].shape[-1]

        if num_betas < 1:
            num_betas = num_total_betas
        else:
            pass

        shapedirs = smpl_dict['shapedirs'][:, :, :num_betas]
        self.comp_register('shapedirs', torch.tensor(shapedirs, dtype=dtype), persistent=persistant_buffer)

        if self.model_type == 'smplx':
            if smpl_dict['shapedirs'].shape[-1] > 300:
                begin_shape_id = 300
            else:
                begin_shape_id = 10
                num_expressions = smpl_dict['shapedirs'].shape[-1] - 10

            exprdirs = smpl_dict['shapedirs'][:, :, begin_shape_id:(begin_shape_id + num_expressions)]
            self.comp_register('exprdirs', torch.tensor(exprdirs, dtype=dtype), persistent=persistant_buffer)

            expression = torch.tensor(np.zeros((1, num_expressions)), dtype=dtype)
            self.comp_register('init_expression', expression, persistent=persistant_buffer)
        else:
            pass

        if self.use_dmpl:
            dmpldirs = np.load(dmpl_file_path)['eigvec']

            dmpldirs = dmpldirs[:, :, :num_dmpls]
            self.comp_register('dmpldirs', torch.tensor(dmpldirs, dtype=dtype), persistent=persistant_buffer)
        else:
            pass

        # Regressor for joint locations given shape - 6890 x 24
        self.comp_register('J_regressor', torch.tensor(smpl_dict['J_regressor'], dtype=dtype), persistent=persistant_buffer)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        if use_posedirs:
            posedirs = smpl_dict['posedirs']
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
            self.comp_register('posedirs', torch.tensor(posedirs, dtype=dtype), persistent=persistant_buffer)
        else:
            self.posedirs = None

        # indices of parents for each joints
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)
        self.comp_register('kintree_table', torch.tensor(kintree_table, dtype=torch.int32), persistent=persistant_buffer)

        # LBS weights
        # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
        weights = smpl_dict['weights']
        self.comp_register('weights', torch.tensor(weights, dtype=dtype), persistent=persistant_buffer)

        self.comp_register('init_trans', torch.zeros((1, 3), dtype=dtype), persistent=persistant_buffer)
        # self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        # root_orient
        # if self.model_type in ['smpl', 'smplh']:
        self.comp_register('init_root_orient', torch.zeros((1, 3), dtype=dtype), persistent=persistant_buffer)

        # pose_body
        if self.model_type in ['smpl', 'smplh', 'smplx']:
            self.comp_register('init_pose_body', torch.zeros((1, 63), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type == 'animal_horse':
            self.comp_register('init_pose_body', torch.zeros((1, 105), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type == 'animal_dog':
            self.comp_register('init_pose_body', torch.zeros((1, 102), dtype=dtype), persistent=persistant_buffer)
        else:
            pass

        # pose_hand
        if self.model_type in ['smpl']:
            self.comp_register('init_pose_hand', torch.zeros((1, 1 * 3 * 2), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type in ['smplh', 'smplx']:
            self.comp_register('init_pose_hand', torch.zeros((1, 15 * 3 * 2), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type in ['mano']:
            self.comp_register('init_pose_hand', torch.zeros((1, 15 * 3), dtype=dtype), persistent=persistant_buffer)
        else:
            pass

        # face poses
        if self.model_type == 'smplx':
            self.comp_register('init_pose_jaw', torch.zeros((1, 1 * 3), dtype=dtype), persistent=persistant_buffer)
            self.comp_register('init_pose_eye', torch.zeros((1, 2 * 3), dtype=dtype), persistent=persistant_buffer)
        else:
            pass

        self.comp_register('init_betas', torch.zeros((1, num_betas), dtype=dtype), persistent=persistant_buffer)

        if self.use_dmpl:
            self.comp_register('init_dmpls', torch.zeros((1, num_dmpls), dtype=dtype), persistent=persistant_buffer)

    def comp_register(self, name, value, persistent=False):
        if sys.version_info[0] > 2:
            self.register_buffer(name, value, persistent)
        else:
            self.register_buffer(name, value)

    def r(self):
        from human_body_prior.tools.omni_tools import copy2cpu as c2c
        return c2c(self.forward().v)

    def forward(self,
                root_orient=None,
                pose_body=None,
                pose_hand=None,
                pose_jaw=None,
                pose_eye=None,
                betas=None,
                trans=None,
                dmpls=None,
                expression=None,
                v_template=None,
                joints=None,
                v_shaped=None,
                return_dict=False,
                **kwargs):
        '''
        前向传播方法接收各种人体参数并计算最终的网格顶点：
        - 参数处理：处理各种可选参数，设置默认值
        - 姿势组合：将各部位姿势组合成完整姿势向量
        - 形状组合：组合beta形状参数和DMPL/表情参数
        - LBS计算：调用线性混合蒙皮(LBS)计算最终顶点位置
        - 结果返回：返回顶点、面、关节位置等信息

        :param root_orient: 根关节朝向
        :param pose_body: 身体姿态
        :param pose_hand: 手部姿态
        :param pose_jaw: 下颌姿态
        :param pose_eye: 眼睛姿态
        :param betas: 形状参数, 控制人体的整体形状特征
        :param expression: 表情参数, 仅适用于SMPL-X模型，控制面部表情变化
        :param dmpls: 动态形状参数, 仅适用于SMPL/SMPL-H模型，捕捉更细微的形状变化
        :param trans: 平移参数, 控制人体在3D空间中的位置
        :param v_template: 模板网格顶点, 可选参数，允许用户提供自定义的模板网格
        :param joints: 关节位置, 可选参数，允许用户提供自定义的关节位置
        :param v_shaped: 变形后的网格顶点, 可选参数，允许用户提供预变形的网格顶点
        :param return_dict: 是否以字典形式返回结果
        :param kwargs: 其他可选参数

        :return: 包含顶点、面、关节位置等信息的字典或对象
        '''
        batch_size = 1

        # compute batchsize by any of the provided variables
        for arg in [root_orient, pose_body, pose_hand, pose_jaw, pose_eye, betas, trans, dmpls, expression, v_template, joints]:
            if arg is not None:
                batch_size = arg.shape[0]
                break

        # assert not (v_template is not None and betas is not None), ValueError('vtemplate and betas could not be used jointly.')
        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'animal_horse', 'animal_dog'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano')
        if root_orient is None:  root_orient = self.init_root_orient.expand(batch_size, -1)

        if self.model_type in ['smplh', 'smpl']:
            if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)
            if pose_hand is None:  pose_hand = self.init_pose_hand.expand(batch_size, -1)
        elif self.model_type == 'smplx':
            if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)
            if pose_hand is None:  pose_hand = self.init_pose_hand.expand(batch_size, -1)
            if pose_jaw is None:  pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
            if pose_eye is None:  pose_eye = self.init_pose_eye.expand(batch_size, -1)
        elif self.model_type in ['mano', ]:
            if pose_hand is None:  pose_hand = self.init_pose_hand.expand(batch_size, -1)
        elif self.model_type in ['animal_horse', 'animal_dog']:
            if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)
        else:
            pass

        if pose_hand is None and self.model_type not in ['animal_horse', 'animal_dog']:
            pose_hand = self.init_pose_hand.expand(batch_size, -1)

        if trans is None: trans = self.init_trans.expand(batch_size, -1)
        if v_template is None: v_template = self.init_v_template.expand(batch_size, -1, -1)
        if betas is None: betas = self.init_betas.expand(batch_size, -1)

        full_pose = None
        if self.model_type in ['smplh', 'smpl']:
            full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=-1)
        elif self.model_type == 'smplx':
            # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
            full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand], dim=-1)
        elif self.model_type in ['mano', ]:
            full_pose = torch.cat([root_orient, pose_hand], dim=-1)
        elif self.model_type in ['animal_horse', 'animal_dog']:
            full_pose = torch.cat([root_orient, pose_body], dim=-1)
        else:
            pass

        if self.use_dmpl:
            if dmpls is None: dmpls = self.init_dmpls.expand(batch_size, -1)
            shape_components = torch.cat([betas, dmpls], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.dmpldirs], dim=-1)
        elif self.model_type == 'smplx':
            if expression is None: expression = self.init_expression.expand(batch_size, -1)
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)
        else:
            shape_components = betas
            shapedirs = self.shapedirs

        """
        LBS 是什么？
        - 线性混合蒙皮（Linear Blend Skinning），也称为骨骼动画（Skeletal Animation） 或刚体蒙皮（Rigid Skinning），
            是计算机图形学和计算机动画中最核心、最广泛应用的技术之一。它用于驱动角色（无论是人、动物还是怪物）的变形和运动。
            
        骨骼（Skeleton）：
        - 这是一个由关节（Joints） 和骨骼（Bones） 组成的层次化结构（通常是一个树形结构，称为“ kinematics tree ”或“ kintree ”）。
        - 每个关节的变换（旋转、平移）会影响到其所有子关节和子骨骼。例如，转动肩膀，整个手臂（肘部、手腕、手）都会跟着动。
        - 在SMPL模型中，这对应着 kintree_table。

        蒙皮网格（Skinned Mesh）：
        - 这是用户最终看到的3D模型表面，由顶点（Vertices）和面（Faces）构成。
        - 在LBS中，网格顶点不直接定义自己的位置，而是通过骨骼来驱动。

        蒙皮权重（Skinning Weights）：
        - 这是LBS的“魔法”所在。每个顶点都关联到一根或多根骨骼，并拥有一个对应的权重（Weight），表示该骨骼对顶点影响力的大小。
        - 权重通常在 [0, 1] 范围内，并且对于一个顶点，所有影响它的骨骼的权重之和为1。
        - 例如，手腕处的一个顶点可能主要受手腕关节影响（权重0.8），但也稍微受小臂关节影响（权重0.2），这使得在手腕弯曲时，变形会更加平滑自然。

        在SMPL模型中，这对应着 weights 矩阵。
        - 绑定姿势（Bind Pose） / T-Pose：
        - 这是骨骼和网格的初始参考姿势。通常是一个易于建模和绑定的姿势，如著名的“T-Pose”。
        - 在这个姿势下，每个关节都有一个逆绑定矩阵（Inverse Bind Matrix），记为 $G_{j}^{-1}$。这个矩阵的作用是：将顶点从世界坐标变换到对应关节的局部空间坐标。
        
        模板网格 v_template
        
        Jtr 指的是“经过姿态与形状变形后的关节在世界坐标系中的三维位置”。更准确地说：
        Jtr = Transformed Joints（或 Joint Transforms Result 中的 joint positions）
        
        verts：
        👉 经过 shape + pose + LBS 之后的 最终网格顶点位置 (B, V, 3)
        Jtr：
        👉 经过 shape + pose + forward kinematics 之后的 关节位置 (B, J, 3)
        """
        verts, Jtr = lbs(
            betas=shape_components,
            pose=full_pose,
            v_template=v_template,
            shapedirs=shapedirs,
            posedirs=self.posedirs,
            J_regressor=self.J_regressor,
            parents=self.kintree_table[0].long(),
            lbs_weights=self.weights,
            joints=joints,
            v_shaped=v_shaped,
            dtype=self.dtype,
        )

        Jtr = Jtr + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)

        res = {}
        res['v'] = verts
        res['f'] = self.f
        res['Jtr'] = Jtr  # Todo: ik can be made with vposer
        # res['bStree_table'] = self.kintree_table

        # if self.model_type == 'smpl':
        #     res['pose_body'] = pose_body
        # elif self.model_type == 'smplh':
        #     res['pose_body'] = pose_body
        #     res['pose_hand'] = pose_hand
        # elif self.model_type == 'smplx':
        #     res['pose_body'] = pose_body
        #     res['pose_hand'] = pose_hand
        #     res['pose_jaw'] = pose_jaw
        #     res['pose_eye'] = pose_eye
        # elif self.model_type in ['mano', 'mano']:
        #     res['pose_hand'] = pose_hand
        res['full_pose'] = full_pose

        if not return_dict:
            class result_meta(object):
                pass

            res_class = result_meta()
            for k, v in res.items():
                res_class.__setattr__(k, v)
            res = res_class
        else:
            pass

        return res
