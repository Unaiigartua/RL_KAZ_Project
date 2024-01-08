from pettingzoo.butterfly import knights_archers_zombies_v10
import glob
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import time
import supersuit as ss

from stable_baselines3 import PPO, DQN
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.vec_env import  VecMonitor
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback





def train_visual(env_fn, iter, type, steps, seed=None, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs)
    env = ss.black_death_v3(env)


    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=40, y_size=40)
        env = ss.frame_stack_v1(env, 3)

    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    log_dir = f"logs/VECTORIZED_FKAZ_{type}_iteration{iter}_{time.strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)

    if iter == 0:

        model = PPO(
            CnnPolicy if visual_observation else MlpPolicy,
            env,
            verbose=3,
            batch_size=256,
            tensorboard_log=log_dir,
            learning_rate=0.0001,

        )
    else:

        if iter == 1: lr = 0.01
        elif iter == 2: lr = 0.005
        elif iter == 3: lr = 0.001


        model = PPO.load(f"model_{type}_v{iter}")
        model.set_env(env)
        model.learning_rate = lr
        model.tensorboard_log = log_dir


    model.learn(total_timesteps=steps, reset_num_timesteps= False)
    iter+=1

    model.save(f"singleTraining/model_knight_v{iter}")

    print(f"Model iteration {iter} has been saved.")

    env.close()


def eval2(env_fn, iter, type, steps, seed=None, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs, render_mode="human")
    env = ss.black_death_v3(env)


    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=40, y_size=40)
        env = ss.frame_stack_v1(env, 3)

    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    log_dir = f"logs/pruebas/VECTORIZED_FKAZ_{type}_iteration{iter}_{time.strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)



    model = PPO.load(f"singleTraining/model_knight_v1")
    model.set_env(env)
    model.learning_rate = 0.001
    model.tensorboard_log = log_dir
    
    model.learn(total_timesteps=steps, reset_num_timesteps= False)
    iter+=1


    env.close()




def eval(env_fn, num_games: int = 100, render_mode = "human", **env_kwargs, ):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
      
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=40, y_size=40)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    model_arch = PPO.load("final_model_knight")
    model_knight = PPO.load("final_model_archer")

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:

                if agent.startswith('archer'):
                    act = model_arch.predict(obs, deterministic=True)[0]
                elif agent.startswith('knight'):
                    act = model_knight.predict(obs, deterministic=True)[0]

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward


env_kwargs_knights =[dict(max_cycles=300, max_zombies=10,  max_arrows= 0, spawn_rate =  8 ,num_archers = 0, num_knights = 4, vector_state=False),
                dict(max_cycles=300, max_zombies=8,  max_arrows= 0, spawn_rate =  10 ,num_archers = 0, num_knights = 4, vector_state=False),
                dict(max_cycles=200, max_zombies=6,  max_arrows= 0, spawn_rate =  15 ,num_archers = 0, num_knights = 4, vector_state=False),
                dict(max_cycles=100, max_zombies=4,  max_arrows= 0, spawn_rate =  20 ,num_archers = 0, num_knights = 4, vector_state=False)]

env_kwargs_archers =[dict(max_cycles=600, max_zombies=10,  max_arrows= 20, spawn_rate =  30 ,num_archers = 3, num_knights = 0, vector_state=False),
                dict(max_cycles=500, max_zombies=8,  max_arrows= 20, spawn_rate =  20 ,num_archers = 3, num_knights = 0, vector_state=False),
                dict(max_cycles=400, max_zombies=6,  max_arrows= 10, spawn_rate =  10 ,num_archers = 3, num_knights = 0, vector_state=False),
                dict(max_cycles=300, max_zombies=4,  max_arrows= 10, spawn_rate =  8 ,num_archers = 3, num_knights = 0, vector_state=False)]

env_kwargs_combined = dict(max_cycles=100, max_zombies=10, max_arrows=10, spawn_rate=10, num_archers=4, num_knights=4,vector_state=False)


env_fn = knights_archers_zombies_v10

steps = [200_000, 1_500_000, 2_500_000, 400_000 ]


#train_visual(env_fn, 0, 'knight', steps=200_000, **env_kwargs_knights[0])


final_kwargs = dict(max_cycles=10000, max_zombies=4,  max_arrows= 10, spawn_rate =  20 ,num_archers = 2, num_knights = 2, vector_state=False)



#eval(env_fn, num_games=1, render_mode="human", **final_kwargs)

eval2(env_fn, 0, 'knight', steps=steps[0], **env_kwargs_knights[2])