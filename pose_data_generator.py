import os

flag_run_raw_pose_processing = False
flag_run_motion_representation = False
flag_run_calculate_mean_variance = True

# ====================================================================================================

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# ====================================================================================================

if not flag_run_raw_pose_processing:
    pass

else:
    from run_raw_pose_processing import main as run_raw_pose_processing

    run_raw_pose_processing()

# ====================================================================================================

if not flag_run_motion_representation:
    pass

else:
    from run_motion_representation import main as run_motion_representation

    run_motion_representation()

# ====================================================================================================

if not flag_run_calculate_mean_variance:
    pass

else:
    from run_calculate_mean_variance import main as run_calculate_mean_variance

    run_calculate_mean_variance()

# ====================================================================================================
