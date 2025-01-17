import copy

configs = dict()

config = dict(
    algo=dict(
        replay_filepath=None,
        batch_T=1,
        batch_B=512,
        learning_rate=1e-3,
        learning_rate_anneal="cosine",  # cosine
        learning_rate_warmup=1000,  # number of updates
        clip_grad_norm=10.0,
        activation_loss_coefficient=0.0,  # rarely if ever use
        validation_split=0.0,
        n_validation_batches=0,  # usually don't do it.
    ),
    encoder=dict(
        channels=None,
        kernel_sizes=None,
        strides=None,
        paddings=None,
        hidden_sizes=None,
        kiaming_init=True,
    ),
    pixel_control_model=dict(  # Needs to match formate of stored values.
        fc_sizes=[256, 32 * 9 * 9],
        reshape=(32, 9, 9),
        kernel_sizes=[4],
        strides=[2],
        channels=[],  # last channels depends on number of actions
        paddings=None,
        output_paddings=None,
        dueling=True,
    ),
    optim=dict(
        weight_decay=0,
    ),
    runner=dict(
        n_updates=int(2e4),  # 20k Usually sufficient for one game?
        log_interval_updates=int(1e3),
    ),
    name="atari_pc",  # probably change this with the filepath
)

configs["atari_pc"] = config
