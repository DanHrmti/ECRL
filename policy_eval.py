import os
import yaml
from pathlib import Path

import isaacgym

import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import ImageSequenceClip

from stable_baselines3.common.env_checker import check_env

from td3_agent import TD3HER

from isaac_panda_push_env import IsaacPandaPush
from isaac_env_wrappers import IsaacPandaPushGoalSB3Wrapper

from utils import load_pretrained_rep_model, load_latent_classifier, check_config

if __name__ == '__main__':
    """
    Script for evaluating trained policies on a given environment.
    """

    #################################
    #            Config             #
    #################################

    model_path = './model_chkpts/<model_path>'
    stat_save_path = None

    # load config files
    config = yaml.safe_load(Path('config/n_cubes/Config.yaml').read_text())
    isaac_env_cfg = yaml.safe_load(Path('config/n_cubes/IsaacPandaPushConfig.yaml').read_text())

    check_config(config, isaac_env_cfg)

    # random seed
    seed = np.random.randint(50000)
    print(f"Random seed: {seed}")

    # output directories
    results_dir = './results'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory {results_dir}")

    #################################
    #         Representation        #
    #################################

    latent_rep_model = load_pretrained_rep_model(dir_path=config['Model']['latentRepPath'], model_type=config['Model']['obsMode'])
    latent_classifier = load_latent_classifier(config, num_objects=isaac_env_cfg["env"]["numObjects"])

    #################################
    #          Environment          #
    #################################

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
    #        Model Analysis       #
    #################################

    model = TD3HER.load(model_path, env,
                        custom_objects=dict(
                            seed=seed,
                            eval_max_episode_length=config['Evaluation'].get('maxEvalEpisodeLen', min(50 * env.num_objects, 200)),
                         ),
                        )

    # evaluate policy in environment
    print(f"Evaluating model {model_path} on {env.num_objects} objects")
    eval_stat_dict = model.evaluate_policy(config["Evaluation"]["numEvalEpisodes"], stat_save_path=stat_save_path)

    # plot goal image
    plt.imshow(eval_stat_dict["goal_img"])
    plt.axis('off')
    plt.show()

    # create episode video
    vid_save_dir = f'{results_dir}/evaluation_episode_video.gif'
    clip = ImageSequenceClip(eval_stat_dict["img_list"], fps=5)
    clip.write_gif(vid_save_dir, fps=5)

    # plot orientation distance distribution
    if env.push_t:
        hist_fig = plt.figure(1, figsize=(5, 5))
        plt.title("Distribution of Orientation Distance from Goal", fontsize=12)
        plt.hist(eval_stat_dict["ori_dist_array"], bins=np.linspace(0, np.pi, num=50), edgecolor='black')
        plt.show()
