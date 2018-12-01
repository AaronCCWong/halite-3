from environment import Environment

class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env):
        current_observation = env.get_observation()
        actions = env.get_actions()