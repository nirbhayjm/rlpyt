import pprint
import sys

from rlpyt.ul.algos.ul_for_rl.cpc import CPC
from rlpyt.ul.experiments.ul_for_rl.configs.dmlab.dmlab_cpc import configs
from rlpyt.ul.models.ul.encoders import DmlabEncoderModel
from rlpyt.ul.runners.unsupervised_learning import UnsupervisedLearning
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.logging.context import logger_context


def build_and_train(
    slot_affinity_code="0slt_1gpu_1cpu",
    log_dir="test",
    run_ID="0",
    config_key="dmlab_cpc",
):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    pprint.pprint(config)

    algo = CPC(
        optim_kwargs=config["optim"],
        EncoderCls=DmlabEncoderModel,
        encoder_kwargs=config["encoder"],
        **config["algo"]
    )
    runner = UnsupervisedLearning(algo=algo, affinity=affinity, **config["runner"])
    name = config["name"]
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
