import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CrossOver-v0',
    entry_point='ma_gym.envs.crossover:CrossOver',
)

register(
    id='CrossOver-v1',
    entry_point='ma_gym.envs.crossover:CrossOver',
    kwargs={'full_observable': True}
)

register(
    id='TrafficJunction-v0',
    entry_point='ma_gym.envs.traffic_junction:TrafficJunction',
)

register(
    id='Combat-v0',
    entry_point='ma_gym.envs.combat:Combat',
)

for game_info in [[(5, 5), 2, 1], [(7, 7), 4, 2]]:
    grid_shape, n_agents, n_preys = game_info
    _game_name = 'PredatorPrey{}x{}'.format(grid_shape[0], grid_shape[1])
    register(
        id='{}-v0'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys
        }
    )
    # fully -observable ( each agent sees observation of other agents)
    register(
        id='{}-v1'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True
        }
    )

    # prey is initialized at random location and thereafter doesn't move
    register(
        id='{}-v2'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys,
            'prey_move_probs': [0, 0, 0, 0, 1]
        }
    )

    # full observability + prey is initialized at random location and thereafter doesn't move
    register(
        id='{}-v3'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True,
            'prey_move_probs': [0, 0, 0, 0, 1]
        }
    )
