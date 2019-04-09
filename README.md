# TextLabs-Agent-Gym
 
Reinforcement Learning environment for training agents on TextLabs games. Adaptation of https://github.com/xingdi-eric-yuan/TextWorld-Coin-Collector, currently using the updated baseline LSTM-DQN model from [First TextWorld Competition](https://competitions.codalab.org/competitions/20865).

For a system overview, refer to our paper:   
[Playing by the Book: An Interactive Game Approach for Action Graph Extraction from Text][ptb_paper].

## Prerequisites
* Python 3
* [PyTorch 0.4][pytorch_install]
* [TextLabs][textlabs_install]
* [tensorboardX][tensorboardx_install]

## Installation

 1. Install prerequisites.
 2. From the folder `textlabs_agent_gym_setup`:
	 - In `textlabs_agent_gym_setup/textlabs_agent_gym/configs.py`, set `root_path` to point to the directory`/path/to/textlabs_agent_gym_setup/textlabs_agent_gym`.
	 - Run `pip install .`
 3. Download Spacy model (for word pre-processing): `python -m spacy download en`

[pytorch_install]: http://pytorch.org/
[textlabs_install]: https://github.com/ronentk/TextLabs
[tensorboardx_install]: https://github.com/lanpa/tensorboardX/
[ptb_paper]: https://arxiv.org/abs/1811.04319

## Usage
From repo root:
1. Generate games for your training and evaluation environments by running `tl-make-lab-games <env_id>`, where `<env_id>` should correspond to the ones at `lstm_dqn_baseline/config.yaml`
2. Run `python lstm_dqn_baseline/train.py lstm_dqn_baseline/configs/config.yaml`. Training progress logs are stored in `exps/<experiment_tag>`

## Experiment Details
For the experiments run in the [paper][ptb_paper], use `lstm_dqn_baseline/configs/config.yaml` with the following training environment ids (best to do this in a multi-cpu environment, can take a while for the harder environments):
 1. `tl_easy_level1_gamesize100_step50_seed54321_{train, validation}`
 2. `tl_easy_level2_gamesize100_step50_seed54321_{train, validation}`
 3. `tl_easy_level6_gamesize100_step50_seed54321_{train, validation}`
 4. `tl_medium_level2_gamesize100_step60_seed54321_{train, validation}`
 5. `tl_hard_level1_gamesize100_step70_seed54321_{train, validation}`
