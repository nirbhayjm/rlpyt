from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import VariantLevel, make_variants

affinity_code = encode_affinity(
    n_cpu_core=16,
    n_gpu=8,
    contexts_per_gpu=2,
    hyperthread_offset=24,
    n_socket=2,
)
runs_per_setting = 3
variant_levels = list()

env_ids = ["Hopper-v3", "HalfCheetah-v3", "Walker2d-v3", "Ant-v3", "Humanoid-v3"]
values = list(zip(env_ids))
dir_names = ["env_{}".format(*v) for v in values]
keys = [("env", "id")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

default_config_key = "ddpg_from_td3_1M_serial"
script = "rlpyt/experiments/scripts/mujoco/qpg/train/mujoco_ddpg_serial.py"
experiment_title = "ddpg_mujoco"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)

default_config_key = "td3_1M_serial"
script = "rlpyt/experiments/scripts/mujoco/qpg/train/mujoco_td3_serial.py"
experiment_title = "td3_mujoco"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)

default_config_key = "sac_1M_serial"
script = "rlpyt/experiments/scripts/mujoco/qpg/train/mujoco_sac_serial.py"
experiment_title = "sac_mujoco"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
