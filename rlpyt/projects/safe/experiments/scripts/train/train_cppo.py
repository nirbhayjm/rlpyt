import pprint
import sys

from rlpyt.projects.safe.cppo_agent import CppoLstmAgent
from rlpyt.projects.safe.cppo_pid import CppoPID
from rlpyt.projects.safe.experiments.configs.cppo_pid import configs
from rlpyt.projects.safe.safety_gym_env import SafetyGymTrajInfo, safety_gym_make
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.logging.context import logger_context


def build_and_train(
    slot_affinity_code="0slt_0gpu_1cpu_1cpr",
    log_dir="test",
    run_ID="0",
    config_key="LSTM",
):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    pprint.pprint(config)

    sampler = CpuSampler(
        EnvCls=safety_gym_make,
        env_kwargs=config["env"],
        TrajInfoCls=SafetyGymTrajInfo,
        **config["sampler"],
    )
    algo = CppoPID(**config["algo"])
    agent = CppoLstmAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"],
    )
    name = "cppo_" + config["env"]["id"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
