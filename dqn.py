import torch.nn as nn
import torch.nn.functional as F


class DQN:
    def __init__(self, map_size, num_actions):
        self.conv1 = nn.Conv2d(map_size, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 5)

        self.fc = nn.Linear(64, 512)
        self.output = nn.Linear(512, num_actions)

    def forward(self, state):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))

        state = state.view(state.size(0), -1)
        state = self.fc(state)
        return self.output(state)
