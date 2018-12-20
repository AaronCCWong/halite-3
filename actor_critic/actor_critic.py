import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.conv = nn.Conv2d(4, 16, 3)
        self.action = nn.Linear(1024, num_actions)
        self.critic = nn.Linear(1024, 1)

    def forward(self, x):
        if x.dim() < 4:
            x = x.unsqueeze(0)
        x = F.relu(self.conv(x))

        x = x.view(x.size(0), -1)
        action_vals = self.action(x)
        critic_val = self.critic(x)

        return F.softmax(action_vals, dim=0), critic_val
