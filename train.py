import argparse, json, subprocess
import logging
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import Adam

from agent import Agent
from dqn import DQN
from environment import Environment
from experiencebuffer import Experience, ExperienceBuffer

parser = argparse.ArgumentParser()
parser.add_argument('--replay-size', type=int, default=100,
                    help='number of replays to buffer (default: 100)')
parser.add_argument('--replay-start', type=int, default=10,
                    help='number of replays to buffer before backprop (default: 5)')
parser.add_argument('--batch-size', type=int, default=8,
                    help='batch size to use for training (default: 16)')
parser.add_argument('--epsilon', type=float, default=0.9,
                    help='epsilon greedy probability (default: 0.2)')
parser.add_argument('--games', type=int, default=10,
                    help='number of games to play (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.90,
                    help='learning rate (default: 0.90)')
args = parser.parse_args()


# map size, leaving fixed for now
map_width = 32
map_height = 32
# arbitrarily chosen constants
num_actions = 5
turns = 2

writer = SummaryWriter()

net = DQN(map_height*map_width, turns, num_actions)
target_net = DQN(map_height*map_width, turns, num_actions)
optimizer = Adam(net.parameters(), lr=args.lr)

buffer = ExperienceBuffer(args.replay_size)
agent = Agent()
env = Environment(map_height, map_width, turns)


def train(args):
    for _ in range(args.games):
        command = "./halite --replay-directory ./replays/ --width 32 --height 32 --no-timeout --results-as-json".split()
        command.append("python3 MyBot1.py")
        command.append("python3 MyBot1.py")

        print(f"Command used:\n{command}")
        results = subprocess.check_output(command)
        results = json.loads(results)
        print(f"Player stats from match:\n{results['stats']}")
    writer.close()


def get_loss(batch, net, target_net, gamma):
    "We will calculate the loss here"
    states, actions, rewards, new_states = batch

    state_action_vs = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    next_states_vs = target_net(new_states).max(1)[0]
    next_states_vs = next_states_vs.detach()

    expected_values = next_states_vs * gamma + rewards
    return F.mse_loss(state_action_vs, expected_values)


if __name__ == "__main__":
    train(args)
