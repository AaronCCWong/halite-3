import logging

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, turns, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(turns * 4 + 1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 5)

        self.fc = nn.Linear(25600, 512)
        self.output = nn.Linear(512, num_actions)

    def forward(self, obs):
        if obs.dim() < 4:
            obs = obs.unsqueeze(0)

        obs = F.relu(self.conv1(obs))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))

        obs = obs.view(obs.size(0), -1)
        obs = F.relu(self.fc(obs))
        return self.output(obs)
