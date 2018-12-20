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
is restricted to one ship. To change the map size, you need to go into the corresponding train.py inside
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

To allow an unrestricted number of ships for each player, you must go into each of the files `MyBot.py`, `MyBot1.py`,
and `MyBot2.py`. In those files you must comment out the lines

```python3
if game.turn_number == 1:
    command_queue.append(me.shipyard.spawn())
```

and uncomment the lines:
```python3
if game.turn_number <= 100 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
    command_queue.append(me.shipyard.spawn())
```

The piece of code that decides the amount of reward to give can be found in the files `MyBot.py`, `MyBot1.py`, and `MyBot2.py`.
To use the reward function that gives rewards based on the scale of halite returned to dock you must comment out:

```python3
if me.shipyard.position == next_pos:
    hal_amount = ship.halite_amount - (0.1 * game_map[ship.position].halite_amount)
    if hal_amount > 100:
        reward = 5.0
    elif hal_amount > 10:
        reward = 0.5
    else:
        reward = -1.0
    elif action == Direction.Still and game_map[next_pos].halite_amount > 0:
        reward = 1.0
    elif action == Direction.Still and game_map[next_pos].halite_amount == 0:
        reward = -1.0
    elif game_map[next_pos].halite_amount > game_map[ship.position].halite_amount:
        reward = 1.0
    else:
        reward = -1.0
```

and uncomment:

```python3
if me.shipyard.position == next_pos:
    halite_deposit_amount = ship.halite_amount - (0.1 * game_map[ship.position].halite_amount)
    reward = -10.0 if halite_deposit_amount <= 100 else halite_deposit_amount
elif 0.1 * game_map[ship.position].halite_amount > ship.halite_amount:
    reward = 0 - max(0.1 * game_map[ship.position].halite_amount, 100.0)
elif action == Direction.Still:
    if ship.position == me.shipyard.position:
        reward = -10.0
    else:
        reward = 0.25 * game_map[ship.position].halite_amount
elif game_map[next_pos].halite_amount > game_map[ship.position].halite_amount:
    reward = 10.0
else:
    reward = 0 - (0.1 * game_map[ship.position].halite_amount)
```

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