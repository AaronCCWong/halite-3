import torch
import numpy as np
from collections import deque, namedtuple


Experience = namedtuple('Experience',
                        field_names=['state', 'action', 'reward', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        random_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, new_states = zip(*[self.buffer[idx] for idx in random_indices])
        return torch.stack(states), torch.stack(actions), torch.stack(rewards), \
               torch.stack(new_states)
