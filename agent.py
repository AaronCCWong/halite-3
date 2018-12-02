import random
from hlt import Direction

class Agent:
    def step(self, net, state, ship, epsilon):
        obs = state.get_observation()
        actions = state.get_actions()

        # use decaying epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            random_action = random.choice(actions)
            return ship.move(random_action)

        action = net(obs)
