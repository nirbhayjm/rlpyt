import copy
import sys

from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import VariantLevel, make_variants

args = sys.argv[1:]
assert len(args) == 2
my_computer = int(args[0])
num_computers = int(args[1])

print(f"MY_COMPUTER: {my_computer},  NUM_COMPUTERS: {num_computers}")

script = (
    "rlpyt/ul/experiments/scripts/rl_with_ul/dmlab/train/dmlab_ppo_rl_with_ul_alt.py"
)

affinity_code = quick_affinity_code(contexts_per_gpu=1, alternating=True)
runs_per_setting = 2
experiment_title = "dmlab_ppo_with_ul_large_baseline_2"

variant_levels_1 = list()
variant_levels_2 = list()
# variant_levels_3 = list()

stop_conv_grads = [False]
ul_update_schedules = ["constant_0"]
min_steps_rl = [0]
values = list(zip(stop_conv_grads, ul_update_schedules, min_steps_rl))
dir_names = ["{}stpcnvgrd_{}_{}minrl".format(*v) for v in values]
keys = [
    ("model", "stop_conv_grad"),
    ("algo", "ul_update_schedule"),
    ("algo", "min_steps_rl"),
]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

stop_conv_grads = [True]
ul_update_schedules = ["constant_2"]
min_steps_rl = [1e5]
values = list(zip(stop_conv_grads, ul_update_schedules, min_steps_rl))
dir_names = ["{}stpcnvgrd_{}_{}minrl".format(*v) for v in values]
keys = [
    ("model", "stop_conv_grad"),
    ("algo", "ul_update_schedule"),
    ("algo", "min_steps_rl"),
]
variant_levels_2.append(VariantLevel(keys, values, dir_names))

min_steps_ul = [2e4]
values = list(zip(min_steps_ul))
dir_names = ["{}minstepul".format(*v) for v in values]
keys = [("algo", "min_steps_ul")]
variant_levels_2.append(VariantLevel(keys, values, dir_names))

ul_pri_alphas = [0.0]
values = list(zip(ul_pri_alphas))
dir_names = ["{}prialpha".format(*v) for v in values]
keys = [("algo", "ul_pri_alpha")]
variant_levels_2.append(VariantLevel(keys, values, dir_names))

levels = [
    # "lasertag_one_opponent_large",
    # "lasertag_three_opponents_small",
    # "rooms_watermaze",
    "explore_goal_locations_large",
]
entropies = [
    # 0.0003,
    # 0.0003,
    # 0.001,
    0.01,
]
values = list(zip(levels, entropies))
dir_names = levels
keys = [("env", "level"), ("algo", "entropy_loss_coeff")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
variant_levels_2.append(VariantLevel(keys, values, dir_names))


variants_1, log_dirs_1 = make_variants(*variant_levels_1)
variants_2, log_dirs_2 = make_variants(*variant_levels_2)

variants = variants_1 + variants_2
log_dirs = log_dirs_1 + log_dirs_2

num_variants = len(variants)
variants_per = num_variants // num_computers

my_start = my_computer * variants_per
if my_computer == num_computers - 1:
    my_end = num_variants
else:
    my_end = (my_computer + 1) * variants_per
my_variants = variants[my_start:my_end]
my_log_dirs = log_dirs[my_start:my_end]

default_config_key = "ppo_ul_16env"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key,),
)
