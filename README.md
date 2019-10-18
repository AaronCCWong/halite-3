# DQN and Actor-Critic applied to Halite

The API for the halite engine is contained in hlt. Some custom modifications were made to this to make
game play and testing easier.

## How to run agents

The executable file halite was built on Ubuntu 18.04 so only works there. If you are trying to run this
on a Mac or Windows OS then you need to go to https://halite.io/learn-programming-challenge/downloads,
download the corresponding python3 starter kit where you will find a prebuilt halite executable. Replace
the one in this directory with that and things should work.

### Commands

This codebase was written in python3.

To run DQN, use:
```bash
python -m dqn_learning.train --games <NUM_OF_EPISODES_TO_RUN>
```

To run actor-critic, use:
```bash
python -m actor_critic.train --games <NUM_OF_EPISODES_TO_RUN>
```

**Note**: The codebase is set up to use the simplified reward function, a 10x10 game map and each player
is restricted to one ship. To change the map size, you need to go into the corresponding `train.py` inside
actor_critic or dq_learning and change this line:

```python3
command = "./halite --replay-directory ./replays/ --width 10 --height 10 --no-timeout --results-as-json".split()
```

to this:
```python3
command = "./halite --replay-directory ./replays/ --width 32 --height 32 --no-timeout --results-as-json".split()
```
You must also make sure to change the variable `map_dim` at the top of the same file and set that to 32. This may
require parameter number changes to the models.

## Displaying collected data

Training data is collected using tensorboardX. Refer to https://github.com/lanpa/tensorboardX
for installation procedures. The data will be written to a folder called `runs` in this directory.
To generate graphs of the data, use the following command in this directory:

```bash
tensorboard --logdir runs
```

## Clearing all training related data
```bash
./clear.sh
```

## Viewing a replay
Replays generated during training are put into the `replays` folder in this directory. The replay files
have the `.hlt` extension. To view the replay, go to https://halite.io/watch-games and drag and drop the
replay file into the box designated for replays.