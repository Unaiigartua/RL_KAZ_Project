from pettingzoo.butterfly import knights_archers_zombies_v10
import glob
import os
import time
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor

#Entrenar con PPO con una Convolutional modo imagenes.


def train(env_fn, route2, route, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
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

    # Wrap the environment with Monitor for logging
    env = VecMonitor(env, log_dir)

    # Use a CNN policy if the observation space is visual
    model = PPO(
        CnnPolicy if visual_observation else MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
        tensorboard_log=log_dir,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}{route}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs, ):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    policy_arch = max(glob.glob(f"{env.metadata['name']}archer*.zip"), key=os.path.getctime)

    policy_knight = max(glob.glob(f"{env.metadata['name']}knight*.zip"), key=os.path.getctime)

    model_arch = PPO.load(policy_arch)
    model_knight = PPO.load(policy_knight)

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


env_fn = knights_archers_zombies_v10

env_kwargs1 = dict(max_cycles=600, max_zombies=10, max_arrows=14, spawn_rate=0, num_archers=2, num_knights=0,
                   vector_state=True)
env_kwargs2 = dict(max_cycles=600, max_zombies=7, max_arrows=12, spawn_rate=20, num_archers=0, num_knights=4,
                   vector_state=False)

env_kwargs3 = dict(max_cycles=100, max_zombies=10, max_arrows=10, spawn_rate=10, num_archers=2, num_knights=2,
                   vector_state=True)

train(env_fn, "pong_training", "knight", steps=1_000_000, seed=117, **env_kwargs2)
# train(env_fn, "pong_training", "archer", steps=200_000, seed=420, **env_kwargs1)
# print("Evaluation")
eval(env_fn, num_games=10, render_mode=None, **env_kwargs2)


# eval(env_fn, num_games=1, render_mode="human", **env_kwargs1)