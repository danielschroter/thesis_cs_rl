from gym.envs.registration import register

register(
    id='beergame-v0',
    entry_point='beer_game.envs:BeerGame',
    kwargs={'demand_dist': "classical"}
)
