from pettingzoo.butterfly import knights_archers_zombies_v10
import glob
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import time
import supersuit as ss

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.vec_env import  VecMonitor
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn import CnnPolicy as dqnCnn


def train_vectorized(env_fn, algo, type, steps, seed=None, **env_kwargs):
    env_kwargs['vector_state'] = True  # Asegurarse de que el entorno use estado vectorizado

    env = env_fn.parallel_env(**env_kwargs)
    env = ss.black_death_v3(env)
    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    log_dir = f"logs/VECTORIZED_FKAZ_{type}_{algo}_{time.strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)

    if(algo == 'ppo'):
        model = PPO(MlpPolicy, env, verbose=3, batch_size=4000, tensorboard_log=log_dir, learning_rate=0.3)
    elif(algo == 'dqn'):
        model = DQN(DQNPolicy, env, verbose=3, batch_size=256, tensorboard_log=log_dir, learning_rate=0.05)
    elif(algo == 'a2c'):
        model = A2C(MlpPolicy, env, verbose=3, tensorboard_log=log_dir, learning_rate=0.05)



    model.learn(total_timesteps=steps, reset_num_timesteps=False)

    model.save(f"model_{type}_vect_{algo}")
    print(f"Model has been saved.")
    env.close()

def train_visual(env_fn, mode, type, steps: int = 10_000, seed: int = 0, **env_kwargs):

    env = env_fn.parallel_env(**env_kwargs)
    env = ss.black_death_v3(env)


    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=40, y_size=40)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    log_dir = f"logs/{mode}/{type}_{time.strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)

    if mode == "dqn":
        model = DQN(
            policy=dqnCnn,
            env=env,
            verbose=3,
            tensorboard_log=log_dir,
            buffer_size=10_000,
            learning_rate=0.001,
            batch_size=32,
            )

    elif mode == "ppo":
        model = PPO(
            CnnPolicy,
            env,
            verbose=3,
            batch_size=256,
            learning_rate=0.001,
            
        )

    elif mode == "a2c":
        model = A2C(
            CnnPolicy if visual_observation else MlpPolicy,
            env,
            verbose=3,
            tensorboard_log=log_dir,
            learning_rate=0.001,
        )

    model.learn(total_timesteps=steps)
    model.save(f"model_{type}_v1")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, model_path, num_games=100, render_mode="human", **env_kwargs):
    # Asegúrate de que estás utilizando el estado vectorizado
    env_kwargs['vector_state'] = True

    # Cargar el modelo entrenado
    model = A2C.load(model_path)  # Cambiar DQN a A2C o el algoritmo utilizado según corresponda

    env = env_fn.parallel_env(render_mode=render_mode, **env_kwargs)
    env = ss.black_death_v3(env)

    # Convertir a entorno vectorizado como durante el entrenamiento
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    obs = env.reset()

    # Intentar acceder a metadata; manejar la excepción si no está disponible
    try:
        env_name = str(env.metadata['name'])
    except AttributeError:
        env_name = "Unknown Environment"

    print(f"\nStarting evaluation on {env_name} (num_games={num_games}, render_mode={render_mode})")

    total_rewards = []
    for _ in range(num_games):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            episode_reward += rewards[0]

            if render_mode == "human":
                env.render()

            done = dones[0]

        total_rewards.append(episode_reward)

    env.close()

    avg_reward = np.mean(total_rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


env_kwargs_knights =[dict(max_cycles=300, max_zombies=10,  max_arrows= 0, spawn_rate =  8 ,num_archers = 0, num_knights = 4, vector_state=False),
                dict(max_cycles=300, max_zombies=8,  max_arrows= 0, spawn_rate =  10 ,num_archers = 0, num_knights = 4, vector_state=False),
                dict(max_cycles=200, max_zombies=6,  max_arrows= 0, spawn_rate =  15 ,num_archers = 0, num_knights = 4, vector_state=False),
                dict(max_cycles=100, max_zombies=4,  max_arrows= 0, spawn_rate =  20 ,num_archers = 0, num_knights = 4, vector_state=False)]

env_kwargs_archers =[dict(max_cycles=600, max_zombies=10,  max_arrows= 20, spawn_rate =  30 ,num_archers = 3, num_knights = 0, vector_state=False),
                dict(max_cycles=500, max_zombies=8,  max_arrows= 20, spawn_rate =  20 ,num_archers = 3, num_knights = 0, vector_state=False),
                dict(max_cycles=400, max_zombies=6,  max_arrows= 10, spawn_rate =  10 ,num_archers = 3, num_knights = 0, vector_state=False),
                dict(max_cycles=300, max_zombies=4,  max_arrows= 10, spawn_rate =  8 ,num_archers = 3, num_knights = 0, vector_state=False)]

env_kwargs_combined = dict(max_cycles=100, max_zombies=10, max_arrows=10, spawn_rate=10, num_archers=2, num_knights=2,vector_state=False)


env_fn = knights_archers_zombies_v10

steps = [60_000, 1_500_000, 2_500_000, 400_000 ]


train_vectorized(env_fn, algo="ppo", type='archer', steps= 1_000_000, **env_kwargs_archers[1])
#train_vectorized(env_fn, algo="dqn", type='archer', steps= 500_000, **env_kwargs_archers[1])
#train_vectorized(env_fn, algo="a2c", type='archer', steps= 500_000, **env_kwargs_archers[1])

eval(env_fn, 'model_archer_vect_ppo', num_games=1, render_mode="human", **env_kwargs_archers[1])