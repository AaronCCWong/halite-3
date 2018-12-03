import logging
import torch
import random

from hlt import Direction

from experiencebuffer import Experience


class Agent:
    def issue_command(self, net, env, ship, epsilon):
        obs = env.get_observation()
        actions = env.get_actions()

        # use decaying epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action_idx = random.choice(range(len(actions)))
            action = actions[action_idx]
            return action_idx, action, ship.move(action)

        q_values = net(obs)
        _, action_idx = torch.max(q_values, dim=1)
        action = actions[action_idx.item()]
        return action_idx.item(), action, ship.move(action)
