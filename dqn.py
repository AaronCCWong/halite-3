import torch.nn as nn
import torch.nn.functional as F


class DQN:
    def __init__(self, map_size, turns, num_actions):
        self.conv1 = nn.Conv3d(map_size * turns * 2, 32, 5)
        self.conv2 = nn.Conv3d(32, 64, 5)
        self.conv3 = nn.Conv3d(64, 64, 5)

        self.fc = nn.Linear(64, 512)
        self.output = nn.Linear(512, num_actions)

    def forward(self, obs):
        obs = F.relu(self.conv1(obs))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))

        obs = obs.view(obs.size(0), -1)
        obs = self.fc(obs)
        return self.output(obs)
