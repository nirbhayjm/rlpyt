import copy

configs = dict()

config = dict(
    agent=dict(
        store_latent=False,  # only if model stop_conv_grad=True
    ),
    algo=dict(
        discount=0.99,
        learning_rate=2.5e-4,
        value_loss_coeff=1.0,
        entropy_loss_coeff=0.01,
        clip_grad_norm=10.0,
        initial_optim_state_dict=None,
        gae_lambda=0.95,
        minibatches=4,
        epochs=4,
        ratio_clip=0.1,
        linear_lr_schedule=True,
        normalize_advantage=False,
        min_steps_rl=0,
        min_steps_ul=0,
        max_steps_ul=None,
        ul_learning_rate=0.001,
        ul_optim_kwargs=None,
        ul_replay_size=1e5,
        ul_update_schedule="constant_1",
        ul_lr_schedule=None,
        ul_lr_warmup=0,
        ul_delta_T=3,
        ul_batch_B=32,
        ul_batch_T=16,
        ul_random_shift_prob=0.1,
        ul_random_shift_pad=4,
        ul_target_update_interval=1,
        ul_target_update_tau=0.01,
        ul_latent_size=256,
        ul_anchor_hidden_sizes=512,
        ul_clip_grad_norm=10.0,
        ul_pri_alpha=0.0,
        ul_pri_beta=1.0,
        ul_pri_n_step_return=1,
    ),
    env=dict(
        game="pong",
        episodic_lives=False,  # new standard
        repeat_action_probability=0.25,  # sticky actions
        horizon=int(27e3),
    ),
    # Will use same args for eval env.
    model=dict(
        hidden_sizes=512,
        stop_conv_grad=False,
        channels=None,
        kernel_sizes=None,
        strides=None,
        kiaming_init=True,
    ),
    optim=dict(),
    runner=dict(
        n_steps=25e6,
        log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=128,
        batch_B=16,
        max_decorrelation_steps=1000,
    ),
)


configs["ppo_ul_16env"] = config
