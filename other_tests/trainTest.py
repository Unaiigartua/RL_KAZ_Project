from pettingzoo.butterfly import knights_archers_zombies_v10
import glob
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


#ENTRENAR CON OTRA ALTERNATIVA A PPO DENTRO DE STABLEbASELINE3


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def train(env_fn, route2, route,  steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    log_dir = f"logs/{env.unwrapped.metadata.get('name')}{route2}_{time.strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(log_dir, exist_ok=True)

    env = VecMonitor(env, log_dir)


    learning_rate = 0.1
    gamma = 0.6
    epsilon = 0.05
    buffer_size = 10000
    batch_size = 256
    target_network_update_freq = 500


    # Neural network architecture
    policy_kwargs = dict(
        net_arch=[64, 64]  # Adjust as needed
    )

    # Create and train the DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_fraction=epsilon,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_interval=target_network_update_freq,
        verbose=1  # Set to 1 for training progress output
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}{route}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq, log_dir):
        super(RewardLoggerCallback, self).__init__()
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.rewards = []

    def _init_callback(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Use env.unwrapped.rewards to get variables from other wrappers
            unwrapped_env = self.model.env.unwrapped
            rewards = getattr(unwrapped_env, 'rewards', None)

            if rewards is None:
                # If rewards attribute is not found in unwrapped environment, print a warning
                print("Warning: 'rewards' attribute not found in unwrapped environment.")
            else:
                self.rewards.append(rewards)

        return True

    def save_rewards(self):
        reward_file = os.path.join(self.log_dir, 'rewards.txt')
        with open(reward_file, 'w') as f:
            for rewards in self.rewards:
                f.write(','.join(map(str, rewards)) + '\n')

def train_visual_custom(env_fn, route2, route,  steps, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    log_dir = f"logs/{env.unwrapped.metadata.get('name')}{route2}_{time.strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)




    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    callback = RewardLoggerCallback(check_freq=100, log_dir=log_dir)

    model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
    model.learn(steps)
    callback.save_rewards()

    #model.save("custom_cnn_ppo_model")


    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()
    env = env_fn.parallel_env(**env_kwargs)
    env = ss.black_death_v3(env)

    num_games = 3

    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)

    env.reset()

    env = VecMonitor(env, log_dir)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

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
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    if agent.startswith('archer'):
                        print("ERRIR")
                    elif agent.startswith('knight'):
                        act = model.predict(obs, deterministic=True)[0]

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



def eval(env_fn,  num_games: int = 100, render_mode: str | None = None,  **env_kwargs, ):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )


    #policy_arch = max(glob.glob(f"{env.metadata['name']}archer*.zip"), key=os.path.getctime)

    #policy_knight = max(glob.glob(f"{env.metadata['name']}knight*.zip"), key=os.path.getctime)


   #model_arch = PPO.load(policy_arch)
    model_knight = PPO.load("custom_cnn_ppo_model")

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
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
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    if agent.startswith('archer'):
                        #act = model_arch.predict(obs, deterministic=True)[0]
                        None
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




env_fn = knights_archers_zombies_v10




# Set vector_state to false in order to use visual observations (significantly longer training time)
env_kwargs1 = dict(max_cycles=600, max_zombies=10,  max_arrows=14, spawn_rate =  0 , num_archers = 2, num_knights = 0, vector_state=True)
env_kwargs2 = dict(max_cycles=600, max_zombies=7,  max_arrows=12, spawn_rate =  20 ,num_archers = 0, num_knights = 4, vector_state=False)

#N value 2knight + 2sword  + 2arch + 10arrw + 10zmb = 26 rows per observation
env_kwargs3 = dict(max_cycles=100, max_zombies=10,  max_arrows=10, spawn_rate =  10 ,num_archers = 2, num_knights = 2, vector_state=True)

train_visual_custom(env_fn, "pong_training", "knight", steps=1000, seed= 11, **env_kwargs2)
#train(env_fn, "pong_training", "archer", steps=200_000, seed=420, **env_kwargs1)
#print("Evaluation")
#eval(env_fn, num_games=10, render_mode=None, **env_kwargs2)


#eval(env_fn, num_games=1, render_mode="human", **env_kwargs2)