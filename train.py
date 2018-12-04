import argparse, json, subprocess
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
                    help='epsilon greedy probability (default: 0.9)')
parser.add_argument('--epsilon-end', type=float, default=0.02,
                    help='epsilon greedy probability (default: 0.02)')
parser.add_argument('--epsilon-decay', type=int, default=10**3,
                    help='amount to decay epsilon by each turn (default: 100)')
parser.add_argument('--games', type=int, default=10,
                    help='number of games to play (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.90,
                    help='learning rate (default: 0.90)')
parser.add_argument('--sync-target', type=int, default=100,
                    help='number of frames before syncing target net (default: 100)')
args = parser.parse_args()


# map size, leaving fixed for now
map_dim = 32
# arbitrarily chosen constants
num_actions = 6
turns = 2

writer = SummaryWriter()

net = DQN(map_dim*map_dim, turns, num_actions)
target_net = DQN(map_dim*map_dim, turns, num_actions)

net.load_state_dict(torch.load('model/model_5.pth'))
target_net.load_state_dict(torch.load('model/model_5.pth'))

optimizer = Adam(net.parameters(), lr=args.lr)

buffer = ExperienceBuffer(args.replay_size)
agent = Agent()
env = Environment(map_dim, map_dim, turns)


def train(args):
    data = {
        'game_num': 0,
        'num_turns_per_game': 300 + 25 * map_dim / 8
    }

    for game_num in range(args.games):
        data['game_num'] = game_num
        with open('data.json', 'w') as outfile:
            json.dump(data, outfile)

        command = "./halite --replay-directory ./replays/ --width 32 --height 32 --no-timeout --results-as-json".split()
        command.append("python3 MyBot.py")
        command.append("python3 MyBot1.py")

        print(f"Command used:\n{command}")
        results = subprocess.check_output(command)
        results = json.loads(results)
        print(f"Player stats from match:\n{results['stats']}")
    writer.close()


def get_loss(batch, net, target_net, gamma):
    "We will calculate the loss here"
    states, actions, rewards, dones, new_states = batch

    state_action_vs = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    next_states_vs = target_net(new_states).max(1)[0]
    next_states_vs[dones] = 0.0
    next_states_vs = next_states_vs.detach()

    expected_values = next_states_vs * gamma + rewards
    return F.mse_loss(state_action_vs, expected_values)


if __name__ == "__main__":
    train(args)
