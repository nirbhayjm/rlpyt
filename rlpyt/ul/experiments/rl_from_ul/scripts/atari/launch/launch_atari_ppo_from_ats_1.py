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
    "rlpyt/ul/experiments/rl_from_ul/scripts/atari/train/atari_ppo_from_ul_serial.py"
)

affinity_code = quick_affinity_code(contexts_per_gpu=3)
runs_per_setting = 3
experiment_title = "ppo_from_ats_1"

variant_levels_1 = list()
# variant_levels_2 = list()
# variant_levels_3 = list()

n_updates = [50e3]
values = list(zip(n_updates))
dir_names = ["{}updates".format(*v) for v in values]
keys = [("pretrain", "n_updates")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

batch_Ts = [1, 16]
batch_Bs = [512, 32]
delta_Ts = [3, 3]
values = list(zip(batch_Ts, batch_Bs, delta_Ts))
dir_names = ["{}T_{}B_{}Dlta".format(*v) for v in values]
keys = [("pretrain", "batch_T"), ("pretrain", "batch_B"), ("pretrain", "delta_T")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

rs_probs = [0.1, 1.0]
values = list(zip(rs_probs))
dir_names = ["{}rsprob".format(*v) for v in values]
keys = [("pretrain", "random_shift_prob")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


# games = ["pong", "qbert", "seaquest", "space_invaders",
#     "alien", "breakout", "frostbite", "gravitar"]
games = ["breakout", "gravitar", "qbert", "space_invaders"]
values = list(zip(games))
dir_names = games
keys = [("env", "game")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))
# variant_levels_3.append(VariantLevel(keys, values, dir_names))


##################################################
# RL CONFIG (mostly)

n_steps = [25e6]
pretrain_algos = ["ATC"]
replays = ["20200608/15M_VecEps_B78"]
model_dirs = ["/data/adam/ul4rl/models/20200904/atari_ats_ul_1/"]
values = list(
    zip(
        n_steps,
        pretrain_algos,
        replays,
        model_dirs,
    )
)
dir_names = ["RlFromUl"]  # TRAIN SCRIPT SPLITS OFF THIS
keys = [
    ("runner", "n_steps"),
    ("pretrain", "algo"),
    ("pretrain", "replay"),
    ("pretrain", "model_dir"),
]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))


stop_conv_grads = [True]
hidden_sizes = [512]
values = list(zip(stop_conv_grads, hidden_sizes))
dir_names = ["{}_stpcnvgrd_{}hs".format(*v) for v in values]
keys = [("model", "stop_conv_grad"), ("model", "hidden_sizes")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))

variants_1, log_dirs_1 = make_variants(*variant_levels_1)
# variants_2, log_dirs_2 = make_variants(*variant_levels_2)

variants = variants_1  # + variants_2
log_dirs = log_dirs_1  # + log_dirs_2

num_variants = len(variants)
variants_per = num_variants // num_computers

my_start = my_computer * variants_per
if my_computer == num_computers - 1:
    my_end = num_variants
else:
    my_end = (my_computer + 1) * variants_per
my_variants = variants[my_start:my_end]
my_log_dirs = log_dirs[my_start:my_end]

default_config_key = "ppo_16env"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key, experiment_title),
)
