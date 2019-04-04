#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

root_path = Path('textlabs_agent_gym_setup/textlabs_agent_gym')
data_path = root_path.parent.parent / 'data'
generated_games_dir = data_path / 'gen_games'
