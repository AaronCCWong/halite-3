import logging
import os, sys
import torch
import random

from hlt import Direction


class Agent:
    def issue_command(self, net, env, epsilon):
        obs = env.get_observation()
        actions = env.get_actions()

        # use decaying epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action_idx = random.choice(range(len(actions)))
            action = actions[action_idx]
            return action_idx, action

        q_values = net(obs)
        _, action_idx = torch.max(q_values, dim=1)
        action = actions[action_idx.item()]
        return action_idx.item(), action
