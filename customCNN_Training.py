
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import torch as th
from torch import nn
import gym
import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
import time
from stable_baselines3.common.vec_env import  VecMonitor
import os



class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def train_custom_cnn(env_fn, steps: int, seed: int = 0, **env_kwargs):
    env = env_fn.parallel_env(render_mode='rgb_array', **env_kwargs)
    env = ss.black_death_v3(env)

    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=40, y_size=40)
        env = ss.frame_stack_v1(env, 3)

    env.reset()

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    log_dir = f"logs/VISUAL_KAZ_archers_CustomCNN{time.strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    tensorboard_log_dir = "./tensorboard_logs/"

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=3,
                tensorboard_log=tensorboard_log_dir, batch_size=256, learning_rate=0.01)

    model.learn(total_timesteps=steps, reset_num_timesteps= False)

    model.save("custom_cnn_ppo_model")
    env.close()




env_fn = knights_archers_zombies_v10



env_kwargs_archers =[dict(max_cycles=600, max_zombies=10,  max_arrows= 20, spawn_rate =  30 ,num_archers = 3, num_knights = 0, vector_state=False),
                dict(max_cycles=500, max_zombies=8,  max_arrows= 20, spawn_rate =  20 ,num_archers = 3, num_knights = 0, vector_state=False),
                dict(max_cycles=400, max_zombies=6,  max_arrows= 10, spawn_rate =  10 ,num_archers = 3, num_knights = 0, vector_state=False),
                dict(max_cycles=300, max_zombies=4,  max_arrows= 10, spawn_rate =  8 ,num_archers = 3, num_knights = 0, vector_state=False)]


train_custom_cnn(env_fn, steps=200_000, seed=3, **env_kwargs_archers[1])

