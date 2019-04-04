"""
Register all environments related to the TextLabs benchmark.
"""
import re
from tw_textlabs import EnvInfos
from gym.envs.registration import register
from textlabs_agent_gym.envs import parse_env_id

PATTERN = re.compile(r"tl_(easy|medium|hard)_level(\d+)_gamesize(\d+)_step(\d+)_seed(\d+)_(train|validation|test)")
SEED_OFFSETS = {'train': 20181216, 'validation': 2372323, 'test': 2938411}
MODE2LEVEL = {"easy": 0, "medium": 10, "hard": 20}


def make(env_id):
    env_dict = parse_env_id(env_id)

    mode = env_dict['mode']
    level = env_dict['level']
    n_games = env_dict['n_games']
    max_steps = env_dict['max_steps']
    random_seed = env_dict['random_seed']
    split = env_dict['split']

    game_generator_seed = SEED_OFFSETS[split] + MODE2LEVEL[mode] * 10000 + level * 1000 + n_games * 100 + random_seed * 10 + max_steps
    env_id = env_id + "-v0"
    register(
        id=env_id,
        entry_point='textlabs_agent_gym.envs:TextLabsGameLevel',
        max_episode_steps=max_steps,
        kwargs={
            'n_games': n_games,
            'level': MODE2LEVEL[mode] + level,
            'generator_seed': game_generator_seed,
            'env_id': env_id,
            'request_infos': EnvInfos(
                objective=True,
                description=True,
                inventory=True,
                command_feedback=True,
                intermediate_reward=True,
                admissible_commands=True,
                verbs=True,
                entities=True,
                facts=True
            )
        }
    )
    return env_id