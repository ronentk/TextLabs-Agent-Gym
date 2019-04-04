#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from textlabs_agent_gym.envs.utils import vocab_from_env
from textlabs_agent_gym.configs import root_path

if __name__ == "__main__":
    save_path = root_path.parent.parent / 'lstm_dqn_baseline' / 'vocab.p'
    env_id = 'tl_medium_level9_gamesize1000_step60_seed11344_train'
    v = vocab_from_env(env_id, save_path)

