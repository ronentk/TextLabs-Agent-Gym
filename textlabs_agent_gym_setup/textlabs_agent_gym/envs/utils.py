#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Mapping, List
from pathlib import Path
import re
from gym.envs.registration import register, spec
import gym
import pickle

import textlabs_agent_gym

PATTERN = re.compile(r"tl_(easy|medium|hard)_level(\d+)_gamesize(\d+)_step(\d+)_seed(\d+)_(train|validation|test)")

def parse_env_id(env_id: str) -> Mapping[str, str]:
    env = {}
    match = re.match(PATTERN, env_id)
    if not match:
        msg = "env_id should match the following pattern:\n{}"
        raise ValueError(msg.format(PATTERN.pattern))

    env['mode'] = match.group(1)
    env['level'] = int(match.group(2))
    env['n_games'] = int(match.group(3))
    env['max_steps'] = int(match.group(4))
    env['random_seed'] = int(match.group(5))
    env['split'] = match.group(6)
    
    return env


def make_batch(env_id, batch_size, parallel=False):
    """ Make an environment that runs multiple games independently.
    Parameters
    ----------
    env_id : str
        Environment ID that will compose a batch.
    batch_size : int
        Number of independent environments to run.
    parallel : {True, False}, optional
        If True, the environment will be executed in different processes.
    """
    env_id = textlabs_agent_gym.make(env_id)
    batch_env_id = "batch{}-".format(batch_size) + env_id
    env_spec = spec(env_id)
    entry_point= 'textlabs_agent_gym.envs:BatchEnv'
    if parallel:
        entry_point = 'textlabs_agent_gym.envs:ParallelBatchEnv'

    register(
        id=batch_env_id,
        entry_point=entry_point,
        max_episode_steps=env_spec.max_episode_steps,
        max_episode_seconds=env_spec.max_episode_seconds,
        nondeterministic=env_spec.nondeterministic,
        reward_threshold=env_spec.reward_threshold,
        trials=env_spec.trials,
        # Setting the 'vnc' tag avoid wrapping the env with a TimeLimit wrapper. See
        # https://github.com/openai/gym/blob/4c460ba6c8959dd8e0a03b13a1ca817da6d4074f/gym/envs/registration.py#L122
        tags={"vnc": "foo"},
        kwargs={'env_id': env_id, 'batch_size': batch_size}
    )

    return batch_env_id


def vocab_from_env(env_id: str, save_path: Path = None) -> List[str]:
    env_id = textlabs_agent_gym.make(env_id)
    env = gym.make(env_id)
    vocab = env.observation_space.vocab
    if save_path:
        with save_path.open(mode='wb') as f:
            pickle.dump( vocab, f)
        
    env.close()
    return vocab
    
def load_vocab(load_path: Path) -> List[str]:
    with load_path.open(mode='rb') as f:
        return pickle.load(f)
    