import argparse
import json
import subprocess
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import Adam


parser = argparse.ArgumentParser()
parser.add_argument('--replay-size', type=int, default=10000,
                    help='number of replays to buffer (default: 1000)')
parser.add_argument('--replay-start', type=int, default=500,
                    help='number of replays to buffer before backprop (default: 150)')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size to use for training (default: 100)')
parser.add_argument('--epsilon', type=float, default=0.9,
                    help='epsilon greedy probability (default: 0.9)')
parser.add_argument('--epsilon-end', type=float, default=0.05,
                    help='epsilon greedy probability (default: 0.1)')
parser.add_argument('--epsilon-decay', type=int, default=10**5,
                    help='amount to decay epsilon by each turn (default: 100)')
parser.add_argument('--games', type=int, default=10,
                    help='number of games to play (default: 10)')
parser.add_argument('--lr', type=float, default=4e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.90,
                    help='learning rate (default: 0.90)')
parser.add_argument('--sync-target', type=int, default=1000,
                    help='number of frames before syncing target net (default: 1000)')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


