from pettingzoo.butterfly import knights_archers_zombies_v10


env = knights_archers_zombies_v10.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
env.close()