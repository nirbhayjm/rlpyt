import os.path as osp
import pprint
import sys

from rlpyt.algos.pg.ppo import PPO
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector

# from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
# from rlpyt.ul.agents.atari_dqn_rl_from_ul_agent import AtariDqnRlFromUlAgent
# from rlpyt.ul.agents.atari_pg_rl_from_ul_agent import AtariPgRlFromUlAgent
from rlpyt.ul.agents.dmlab_pg_agent import DmlabPgLstmAlternatingAgent

# from rlpyt.envs.atari.atari_env import AtariTrajInfo
# from rlpyt.adam.atari_env import AtariEnv84
from rlpyt.ul.envs.dmlab import DmlabEnv
from rlpyt.ul.experiments.rl_from_ul.configs.dmlab_ppo_from_ul import configs
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.logging.context import logger_context


def build_and_train(
    slot_affinity_code="0slt_1gpu_1cpu",
    log_dir="test",
    run_ID="0",
    config_key="ppo_16env",
    experiment_title="exp",
    snapshot_mode="none",
    snapshot_gap=None,
):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    # Hack that the first part of the log_dir matches the source of the model
    model_base_dir = config["pretrain"]["model_dir"]
    if model_base_dir is not None:
        raw_log_dir = log_dir.split(experiment_title)[-1].lstrip(
            "/"
        )  # get rid of ~/GitRepos/adam/rlpyt/data/local/<timestamp>/
        model_sub_dir = raw_log_dir.split("/RlFromUl/")[
            0
        ]  # keep the UL part, which comes first
        config["agent"]["state_dict_filename"] = osp.join(
            model_base_dir, model_sub_dir, "run_0/params.pkl"
        )
    pprint.pprint(config)

    sampler = AlternatingSampler(
        EnvCls=DmlabEnv,
        env_kwargs=config["env"],
        CollectorCls=GpuWaitResetCollector,
        # TrajInfoCls=AtariTrajInfo,
        # eval_env_kwargs=config["env"],  # Same args!
        **config["sampler"]
    )
    algo = PPO(optim_kwargs=config["optim"], **config["algo"])
    agent = DmlabPgLstmAlternatingAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo, agent=agent, sampler=sampler, affinity=affinity, **config["runner"]
    )
    name = config["env"]["level"]
    if snapshot_gap is not None:
        snapshot_gap = int(snapshot_gap)
    with logger_context(
        log_dir,
        run_ID,
        name,
        config,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
    ):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
