"""
Main
Author: Dan Haramati
"""

import os
import time
import yaml
from pathlib import Path
import argparse

import isaacgym

import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise

import wandb

from td3_agent import TD3HER
from policies import CustomTD3Policy, EITActor, EITCritic, SMORLActor, SMORLCritic, MLPActor, MLPCritic

from isaac_panda_push_env import IsaacPandaPush
from isaac_env_wrappers import IsaacPandaPushGoalSB3Wrapper

from multi_her_replay_buffer import MultiHerReplayBuffer

from utils import load_pretrained_rep_model, load_latent_classifier, check_config, get_run_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ECRL Training")
    parser.add_argument("-c", "--config_dir", type=str, default='config/n_cubes', help="path to config files")
    args = parser.parse_args()

    #################################
    #            Config             #
    #################################

    config_dir = args.config_dir

    # load config files
    config = yaml.safe_load(Path(f'{config_dir}/Config.yaml').read_text())
    isaac_env_cfg = yaml.safe_load(Path(f'{config_dir}/IsaacPandaPushConfig.yaml').read_text())
    policy_config = yaml.safe_load(Path('config/PolicyConfig.yaml').read_text())

    check_config(config, isaac_env_cfg, policy_config)

    ############ WANDB #############
    if config['WANDB']['log']:
        # initialize weights & biases run
        wandb_run = wandb.init(
            entity="",
            project="ECRL",
            sync_tensorboard=False,
            settings=wandb.Settings(start_method="fork"),
        )
    #################################

    # random seed
    seed = np.random.randint(50000)
    print(f"\nRandom seed: {seed}")

    # run name
    name = get_run_name(config, isaac_env_cfg, seed)
    if config['WANDB']['log']:
        wandb_run.name = name
    print(f"Run Name: {name}")

    # output directories
    results_dir = './results'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory {results_dir}")

    models_dir = './model_chkpts'
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory {models_dir}")

    model_save_dir = models_dir + f'/model_{name}_{seed}'

    #################################
    #         Representation        #
    #################################

    latent_rep_model = load_pretrained_rep_model(dir_path=config['Model']['latentRepPath'], model_type=config['Model']['obsMode'])
    latent_classifier = load_latent_classifier(config, num_objects=isaac_env_cfg["env"]["numObjects"])

    #################################
    #          Environment          #
    #################################

    print(f"Setting up environment...")

    # create environments
    envs = IsaacPandaPush(
        cfg=isaac_env_cfg,
        rl_device=f"cuda:{config['cudaDevice']}",
        sim_device=f"cuda:{config['cudaDevice']}",
        graphics_device_id=config['cudaDevice'],
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    # wrap enviroments for GoalEnv and SB3 compatibility
    env = IsaacPandaPushGoalSB3Wrapper(
        env=envs,
        obs_mode=config['Model']['obsMode'],
        n_views=config['Model']['numViews'],
        latent_rep_model=latent_rep_model,
        latent_classifier=latent_classifier,
        reward_cfg=config['Reward']['GT'],
        smorl=(config['Model']['method'] == 'SMORL'),
    )

    if config['envCheck']:
        check_env(env, warn=True, skip_render_check=True)  # verify SB3 compatibility

    print(f"Finished setting up environment")

    #################################
    #            Policy             #
    #################################

    policy_kwargs = policy_config[config['Model']['method']][config['Model']['obsType']]

    if config['Model']['method'] == 'ECRL':
        policy_kwargs['actor_class'] = EITActor
        policy_kwargs['critic_class'] = EITCritic

    elif config['Model']['method'] == 'SMORL':
        policy_kwargs['actor_class'] = SMORLActor
        policy_kwargs['critic_class'] = SMORLCritic

    elif config['Model']['method'] == 'Unstructured':
        policy_kwargs['actor_class'] = MLPActor
        policy_kwargs['critic_class'] = MLPCritic

    else:
        raise NotImplementedError(f"Method type '{config['Model']['method']}' is not supported")

    #################################
    #        Model & Training       #
    #################################

    # Training parameters
    epoch_episodes = config['Training']['epochEpisodes']
    epoch_timesteps = config['Training']['epochEpisodes'] * env.horizon
    total_timesteps = ((config['Training']['totalTimesteps'][env.num_objects-1] // epoch_timesteps) + 1) * epoch_timesteps
    if config['collectData']:
        total_timesteps = config['collectDataNumTimesteps']

    # Exploration Params
    # Action noise
    action_dim = env.action_space.shape[-1]
    action_noise_sigma = config['Training']['actionNoiseSigma']
    action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=action_noise_sigma * np.ones(action_dim))

    # Epsilon greedy
    exploration_epsilon = config['Training']['explorationEpsilon']

    # Noise schedule: [fraction of initial value at end of schedule, episode start decay, episode end decay]
    if env.push_t:
        exploration_schedule = [0.5, 30 * epoch_episodes, 40 * epoch_episodes]
    elif env.num_objects == 1:
        exploration_schedule = [0.5, 10 * epoch_episodes, 20 * epoch_episodes]
    elif env.num_objects == 2:
        exploration_schedule = [0.5, 20 * epoch_episodes, 30 * epoch_episodes]
    elif env.num_objects == 3:
        exploration_schedule = [0.5, 30 * epoch_episodes, 40 * epoch_episodes]
    else:
        exploration_schedule = [0.5, 30 * epoch_episodes, 40 * epoch_episodes]  # default

    # Model
    model = TD3HER(
        env=env,  # wrapped IsaacGym environment
        policy=CustomTD3Policy,  # policy class
        policy_kwargs=policy_kwargs,  # policy and Q-function related parameters
        learning_rate=config['Training']['learningRate'],  # learning rate for agent Adam optimizer
        batch_size=config['Training']['batchSize'],
        tau=config['Training']['tau'],  # soft update coefficient
        gamma=config['Training']['gamma'],  # discount factor
        a_reg_coef=config['Training']['actionRegCoefficient'],  # action regularization coefficient for actor loss
        buffer_size=min(total_timesteps, config['Training']['bufferSize'][env.num_objects-1]) if not config["collectData"] else total_timesteps,  # size of the replay buffer
        replay_buffer_class=MultiHerReplayBuffer,  # use TD3 with HER
        replay_buffer_kwargs=dict(  # HER parameters:
            n_sampled_goal=0 if env.ordered_push else 4,  # real-to-relabled transition ratio
            goal_selection_strategy="future",  # sample from states after current in same episode
            online_sampling=True,  # sample a new goal with each minibatch
            max_episode_length=env.horizon,  # maximum number of steps in episode
            handle_timeout_termination=True,  # removes termination signals due to timeout
        ),
        chamfer_reward=config['Model']['ChamferReward'],  # if False, use GT reward from environment
        chamfer_reward_kwargs=config['Reward']['Chamfer'],
        learning_starts=(config['Training']['warmupEpisodes'] * env.horizon) if not config['collectData'] else total_timesteps,  # how many steps of the model to collect transitions for before learning starts
        train_freq=(env.horizon, "step"),  # frequency for model update, should determine choice of number of parallel envs
        gradient_steps=int(env.num_envs * env.horizon * config['Training']['utdRatio']),  # gradient steps per train_freq (default=-1, 1 gradient step for each env step)
        action_noise=action_noise,  # initial action noise
        exploration_epsilon=exploration_epsilon,  # initial probability for uniform action
        exploration_schedule=exploration_schedule,
        policy_eval_freq=epoch_episodes if not config['collectData'] else None,  # frequency in episodes to evaluate policy on
        num_eval_episodes=config['Evaluation']['numEvalEpisodes'],  # number of episodes to evaluate policy on
        eval_max_episode_length=config['Evaluation'].get('maxEvalEpisodeLen', 50*env.num_objects),
        smorl_meta_n_steps=config['Evaluation']['SMORLMetaNumSteps'],  # number of consecutive steps attempting to solve each SMORL goal before cycling to the next goal
        model_save_freq=epoch_episodes,
        model_save_dir=model_save_dir,
        seed=seed,
        device=f"cuda:{config['cudaDevice']}",  # "auto" = use GPU if available
        _init_setup_model=True,  # build the network at the creation of the instance
        wandb_log=config['WANDB']['log'],
        wandb_log_policy_stats=config['WANDB']['logPolicyStats'],  # setting to False saves a lot of time in training
        episode_vis_freq=((config['WANDB']['episodeVisFreq'] // env.num_envs) * env.num_envs),  # frequency in episodes to visualize policy on WANDB
    )

    ########## WANDB ##########
    if config['WANDB']['log']:

        # Hyper-parameters
        wandb.config.update(dict(
            seed=model.seed,
            obs_mode=model.obs_mode,
            lr=model.learning_rate,
            bs=model.batch_size,
            tau=model.tau,
            gamma=model.gamma,
            a_reg_coef=model.a_reg_coef,
            action_noise=model.an_sigma_init,
            exp_epsilon=model.epsilon_init,
            buffer_size=model.buffer_size,
            her_ratio=model.replay_buffer.her_ratio,

            epoch_episodes=epoch_episodes,
            horizon=env.horizon,
            warmup_episodes=config['Training']['warmupEpisodes'],
            total_timesteps=total_timesteps,
            utd_ratio=config['Training']['utdRatio'],

            reward_scale=env.reward_scale,
            chamfer_reward=config['Model']['ChamferReward'],
        ))

        if config['Model']['obsMode'] not in ['vae', 'state_unstruct']:
            wandb.config.update(dict(
                # Policy related parameters
                h_dim=policy_config[config['Model']['method']][config['Model']['obsType']]["actor_kwargs"]["h_dim"],
                embed_dim=policy_config[config['Model']['method']][config['Model']['obsType']]["actor_kwargs"]["embed_dim"],
                n_head=policy_config[config['Model']['method']][config['Model']['obsType']]["actor_kwargs"]["n_head"],
                masking=policy_config[config['Model']['method']][config['Model']['obsType']]["actor_kwargs"]["masking"] if config['Model']['obsMode'] == "dlp" else None,
            ))

        if config['Model']['obsMode'] in ['dlp', 'vae', 'slot']:
            wandb.config.update(dict(latent_rep_chkpt=config['Model']['latentRepPath']))

        # Tags
        wandb_run.tags = wandb_run.tags + (f"{env.num_objects}Obj",)
        if env.num_objects != env.num_colors:
            wandb_run.tags = wandb_run.tags + (f"{env.num_colors}Color",)

        if config['Model']['ChamferReward']:
            wandb_run.tags = wandb_run.tags + ('ChamferReward',)

        for key in ["AdjacentGoals", "OrderedPush", "PushT", "RandColor", "RandNumObj"]:
            if key in isaac_env_cfg["env"] and isaac_env_cfg["env"][key]:
                wandb_run.tags = wandb_run.tags + (key,)

        if env.small_table:
            wandb_run.tags = wandb_run.tags + ("SmallTable", )

    ###########################

    # training
    print("\nTraining started")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, log_interval=None)
    env.close()
    print("Training finished")
    print(f"Elapsed time: {(time.time() - start_time) / 3600:5.2f}h")

    # post Training
    print(f"Saving model to {model_save_dir}")
    model.save(model_save_dir)

    if config["collectData"]:  # save media to pretrain representation model
        data_save_dir = results_dir + f'/{env.num_objects}C_{total_timesteps}ts_res{isaac_env_cfg["env"]["cameraRes"]}'
        print(f"Saving image data to {data_save_dir}")
        info = model.replay_buffer.info_buffer.copy()
        total_episodes = len(info) * env.num_envs  # assumes size of replay buffer is <= amount of data collected
        obs_shape = info[0][0][0]['image'].shape
        transitions = np.zeros(np.append([total_episodes, model.horizon], obs_shape), dtype=np.uint8)
        for e in range(len(info)):
            for i in range(model.horizon):
                for env_idx in range(env.num_envs):
                    transitions[env.num_envs * e + env_idx][i] = info[e][i][env_idx]['image']
        np.save(data_save_dir, transitions)
