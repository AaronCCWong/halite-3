import logging

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, turns, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(turns * 4, 16, 3)

        self.fc = nn.Linear(1024, 512)
        self.output = nn.Linear(512, num_actions)

    def forward(self, obs):
        if obs.dim() < 4:
            obs = obs.unsqueeze(0)

        obs = F.relu(self.conv1(obs))

        obs = obs.view(obs.size(0), -1)
        obs = F.relu(self.fc(obs))
        return self.output(obs)
