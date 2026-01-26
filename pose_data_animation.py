# ============================================
# Parallel Human Motion Renderer (Matplotlib)
# ============================================

import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ------------------------------------------------
# IMPORTANT: use headless backend BEFORE pyplot
# ------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =================================================
# Render Configuration
# =================================================

class RenderConfig:
    FPS = 50
    RADIUS = 4.0
    FIGSIZE = (8, 8)
    CAMERA_ELEV = 120
    CAMERA_AZIM = -90
    CAMERA_DIST = 7.5


# =================================================
# Motion Normalization
# =================================================

class MotionNormalizer:
    @staticmethod
    def normalize(joints: np.ndarray):
        """
        Args:
            joints: (T, J, 3)
        Returns:
            normalized_joints, trajectory(T,2), mins, maxs
        """
        assert joints.ndim == 3 and joints.shape[-1] == 3

        data = joints.copy()
        mins = data.min(axis=(0, 1))
        maxs = data.max(axis=(0, 1))

        # ground alignment
        data[:, :, 1] -= mins[1]

        # root trajectory
        traj = data[:, 0, [0, 2]]

        # root-centered
        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]

        return data, traj, mins, maxs


# =================================================
# Animator (single file, single process)
# =================================================

class MotionAnimator:
    def __init__(self, joints, kinematic_tree, title=""):
        self.joints, self.traj, self.mins, self.maxs = \
            MotionNormalizer.normalize(joints)

        self.frames = joints.shape[0]
        self.kinematic_tree = kinematic_tree

        self.fig = plt.figure(figsize=RenderConfig.FIGSIZE)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.suptitle(title)

        self._setup_camera()
        self._init_artists()

    def _setup_camera(self):
        ax = self.ax
        ax.view_init(RenderConfig.CAMERA_ELEV, RenderConfig.CAMERA_AZIM)
        ax.dist = RenderConfig.CAMERA_DIST
        ax.set_xlim3d([-RenderConfig.RADIUS / 2, RenderConfig.RADIUS / 2])
        ax.set_ylim3d([0, RenderConfig.RADIUS])
        ax.set_zlim3d([0, RenderConfig.RADIUS])
        ax.axis("off")

    def _init_artists(self):
        # skeleton lines
        self.lines = []
        colors = ['red', 'blue', 'black', 'darkred', 'darkblue']

        for chain, color in zip(self.kinematic_tree, colors):
            line, = self.ax.plot([], [], [], color=color, linewidth=3.0)
            self.lines.append((chain, line))

        # trajectory line
        self.traj_line, = self.ax.plot([], [], [], color='blue', linewidth=1.0)

        # ground plane (created once)
        verts = [[
            [self.mins[0], 0, self.mins[2]],
            [self.mins[0], 0, self.maxs[2]],
            [self.maxs[0], 0, self.maxs[2]],
            [self.maxs[0], 0, self.mins[2]],
        ]]
        self.ground = Poly3DCollection(verts, alpha=0.5, facecolor='gray')
        self.ax.add_collection3d(self.ground)

    def _update(self, frame):
        # skeleton
        for chain, line in self.lines:
            chain = [j for j in chain if j < self.joints.shape[1]]
            if len(chain) < 2:
                continue

            xs = self.joints[frame, chain, 0]
            ys = self.joints[frame, chain, 1]
            zs = self.joints[frame, chain, 2]

            line.set_data(xs, ys)
            line.set_3d_properties(zs)

        # trajectory
        if frame > 1:
            self.traj_line.set_data(
                self.traj[:frame, 0] - self.traj[frame, 0],
                np.zeros(frame)
            )
            self.traj_line.set_3d_properties(
                self.traj[:frame, 1] - self.traj[frame, 1]
            )

        # ground translation
        dx, dz = self.traj[frame]
        verts = [[
            [self.mins[0] - dx, 0, self.mins[2] - dz],
            [self.mins[0] - dx, 0, self.maxs[2] - dz],
            [self.maxs[0] - dx, 0, self.maxs[2] - dz],
            [self.maxs[0] - dx, 0, self.mins[2] - dz],
        ]]
        self.ground.set_verts(verts)

        return []

    def save(self, path):
        anim = FuncAnimation(
            self.fig,
            self._update,
            frames=self.frames,
            interval=1000 / RenderConfig.FPS,
            repeat=False
        )
        anim.save(path, fps=RenderConfig.FPS)
        plt.close(self.fig)


# =================================================
# Worker (one file per process)
# =================================================

def render_one_motion(args):
    npy_path, save_path, kinematic_tree = args

    if os.path.exists(save_path):
        return

    joints = np.load(npy_path)
    animator = MotionAnimator(joints, kinematic_tree)
    animator.save(save_path)


# =================================================
# Parallel Entry Point
# =================================================

def main():
    src_dir = "./HumanML3D/new_joints/"
    tgt_dir = "./HumanML3D/animations/"
    os.makedirs(tgt_dir, exist_ok=True)

    kinematic_tree = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]

    tasks = []
    for file in sorted(os.listdir(src_dir)):
        if not file.endswith(".npy"):
            continue

        npy_path = os.path.join(src_dir, file)
        save_path = os.path.join(tgt_dir, file.replace(".npy", ".mp4"))
        tasks.append((npy_path, save_path, kinematic_tree))

    num_workers = min(cpu_count(), 8)  # 建议 4–8
    print(f"[INFO] Using {num_workers} parallel workers")

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(render_one_motion, tasks),
                  total=len(tasks)))


# =================================================
# Main Guard (REQUIRED)
# =================================================

if __name__ == "__main__":
    main()
