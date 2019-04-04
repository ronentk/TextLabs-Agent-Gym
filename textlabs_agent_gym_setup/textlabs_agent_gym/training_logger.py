#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import shutil
from typing import Optional, List, Any, Mapping
import time
import yaml
import logging.config
import json
from os.path import join as pjoin
from tensorboardX import SummaryWriter
import jsonpickle
import numpy as np

class Episode:
    def __init__(self, split: str, slot_num: int, epoch_num: int, batch_num: int,
                 objectives: Optional[List[str]] = [],
                 acts: Optional[List[str]] = [], 
                 obs: Optional[List[str]] = [], 
                 reward: Optional[List[str]] = [], 
                 int_reward: Optional[List[str]] = [],
                 game_file: str = None):
        self.split = split
        self.slot_num = slot_num
        self.epoch_num = epoch_num
        self.batch_num = batch_num
        self.objectives = objectives if objectives else []
        self.acts = acts if acts else []
        self.obs = obs if obs else []
        self.rwrds = reward if reward else []
        self.int_rwrds = int_reward if int_reward else []
        self.game_file = game_file
    
    def __str__(self):
        return jsonpickle.encode(self)
    
    def reward(self) -> int:
        # at each timestep is the cumulative reward until that point
        return max(self.rwrds)  
    
    def end_step(self) -> int:
        # return step at which game ended- in our case, when the reward
        # was received (or not)
        # TODO should be by 'dones'
        try:
            end_step = self.rwrds.index(1)
        except ValueError:
            end_step = len(self.rwrds)
        return end_step
    
    def intermediate_reward(self) -> bool:
        int_rewards = np.array(self.int_rwrds)
        return np.sum(int_rewards[:self.end_step()])
        
        
    
    
    

class BatchEpisode:
    def __init__(self, epoch_num: int, batch_num: int, name: str, 
                 objectives: Optional[List[str]] = [],
                 acts: Optional[List[str]] = [], 
                 obs: Optional[List[str]] = [], 
                 reward: Optional[List[str]] = [], 
                 int_reward: Optional[List[str]] = [],
                 extra_infos: Optional[Mapping[str, Any]] = {}):
        self.epoch_num = epoch_num
        self.batch_num = batch_num
        self.name = name
        self.objectives = objectives if objectives else []
        self.acts = acts if acts else []
        self.obs = obs if obs else []
        self.rwrds = reward if reward else []
        self.int_rwrds = int_reward if int_reward else []
        self.extra_infos = extra_infos if extra_infos else {}
    
    def to_episodes(self):
        episodes = []
        acts_by_episode = list(map(list, zip(*self.acts)))
        obs_by_episode = list(map(list, zip(*self.obs)))
        rewards_by_episode = list(map(list, zip(*self.rwrds)))
        int_rewards_by_episode = list(map(list, zip(*self.int_rwrds)))
        num_eps_per_batch = len(acts_by_episode)
        objs_by_episode = [self.objectives[i] if self.objectives else \
                           None for i in range(num_eps_per_batch)]
        if ('game_file' in self.extra_infos and 
            (len(self.extra_infos['game_file']) == num_eps_per_batch)):
            game_files_by_episode = self.extra_infos['game_file']
        else:
            game_files_by_episode = [None] * num_eps_per_batch
        for i in range(num_eps_per_batch):
            episodes.append(Episode(self.name, i,self.epoch_num, self.batch_num,
                                    objs_by_episode[i],
                                    acts_by_episode[i],
                                    obs_by_episode[i],
                                    rewards_by_episode[i],
                                    int_rewards_by_episode[i],
                                    game_files_by_episode[i]))
        return episodes
    
    def win_episodes(self, by_obs=True):
        episodes = self.to_episodes()
        win_eps = []
        lose_eps = []
        for ep in episodes:
            if by_obs:
                all_obs = ' '.join(ep.obs)
                if 'score' in all_obs:
                    win_eps.append(ep)
                else:
                    lose_eps.append(ep)
            else:
                if any(ep.rwrds):
                    win_eps.append(ep)
                else:
                    lose_eps.append(ep)
                    
        return win_eps, lose_eps
            
        
        
        
    def __str__(self):
        return jsonpickle.encode(self)
    

    def update(self, acts: List[str], obs: List[str], reward: List[int], 
                 int_reward: Optional[List[int]],
                 extra_infos: Optional[Mapping[str, Any]]):
        self.acts.append(acts)
        self.obs.append(obs)
        self.rwrds.append(reward)
        if int_reward:
            self.int_rwrds.append(int_reward)
        if extra_infos:
            self.extra_infos.update(extra_infos)

def load_log(logfile_path: Path) -> List[BatchEpisode]:
    eps = []
    for line in logfile_path.open():
        try:
            eps.append(jsonpickle.decode(line))
        except:
            pass
    return eps
        
class GameLogger:
    def __init__(self, path: str, batches_per_epoch: int, log_every_x_batches: int = 1, 
                 name: str = "train"):
        self._path = Path(path) / 'episodes_{}.elg'.format(name)
        self._path.write_text('Logging started.')
        # for every call <log_every> to log_step, perform call once.
        self.batches_per_epoch = batches_per_epoch
        self._log_every_x_batches = log_every_x_batches  
        self._step_counter = 0
        self.total_batches = 0
        self._episodes = []
        self.current_ep = BatchEpisode(epoch_num=self.epoch_counter, batch_num=self.batch_counter,
                                       name=name)
        self.name = name
        
    @property
    def logging(self):
        return (self.batch_counter % self._log_every_x_batches) == 0
    
    @property
    def epoch_counter(self):
        return self.total_batches // self.batches_per_epoch
    
    @property
    def batch_counter(self):
        return self.total_batches % self.batches_per_epoch
    
    def reset(self, objectives: Optional[List[str]] = [], 
              game_files: Optional[List[str]] = []):
        self.current_ep = BatchEpisode(self.epoch_counter, self.batch_counter,
                                       self.name,
                                  objectives=objectives, 
                                  extra_infos={'game_file': game_files})
    
    def log_step(self, acts: List[str], obs: List[str], reward: List[int], 
                 int_reward: Optional[List[int]], 
                 extra_infos: Optional[Mapping[str, Any]] = {}):
        if not self.logging:
            return
        int_reward = [int(x) for x in int_reward]
        self.current_ep.update(acts, obs, reward, int_reward, extra_infos)
    
    def end_episode(self):
        if self.logging:
            self._episodes.append(self.current_ep)
            with self._path.open(mode='a') as f:
                f.write("\n" + str(self.current_ep))
        self.total_batches += 1
                

######## 
# From https://github.com/xingdi-eric-yuan/TextWorld-Coin-Collector/blob/master/helpers/setup_logger.py

def setup_logging(
        default_config_path='config/logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG',
        add_time_stamp=False,
        default_logs_path='./'
):
    """Setup logging configuration

    """
    path = default_config_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())

        if add_time_stamp:
            add_time_2_log_filename(config)

        # Create logs folder if needed.
        for handler in config["handlers"].values():
            if "filename" in handler:
                handler["filename"] = pjoin(default_logs_path, handler["filename"])
                dirname = os.path.dirname(handler["filename"])
                try:
                    os.makedirs(dirname)
                except FileExistsError:
                    pass

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    
    return config


def add_time_2_log_filename(config):
    for k, v in config.items():
        if k == 'filename':
            config[k] = v + "." + time.strftime("%Y-%d-%m-%s")
            print('log file name: %s' % config[k])
        elif type(v) is dict:
            add_time_2_log_filename(v)


def goal_prompt(logger, prompt='What are you testing in this experiment? '):
    print("            ***************************")
    goal = input(prompt)
    logger.info("            ***************************")
    logger.info("TEST GOAL: %s" % goal)


def log_git_commit(logger):
    try:
        commit = get_git_revision_hash()
        logger.info("current git commit: %s" % commit)
    except:
        logger.info('cannot get git commit.')


def get_git_revision_hash():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

##################
    

class TrainingLogger:
    
    def __init__(self, exp_dir: str, configs_dir: str, batches_per_epoch: int, 
                 log_every_x_batches: Optional[int] = None,
                 eval_batches_per_epoch: Optional[int] = None, 
                 eval_log_every_x_batches: Optional[int] = None,
                 ):
        self.exp_dir = Path(exp_dir)
        self.configs_dir = Path(configs_dir)
        self.env_logger = logging.getLogger(__name__)
        self._curr_epoch = 0

        # setup env logger
        logging_config = setup_logging(
            default_config_path= str(self.configs_dir / 'logging_config.yaml'),
            default_level=logging.INFO,
            env_key='LOG_CFG',
            add_time_stamp=True,
            default_logs_path=self.exp_dir
        )
        
        curr_git_hash = get_git_revision_hash().decode("utf-8").strip()
        
        self.logs_dir = self.exp_dir / logging_config['log_dir']
        
        run_config_file = self.configs_dir / 'config.yaml'
        with run_config_file.open() as f:
            self.run_config = yaml.load(f.read())
            self.run_config['git_hash'] = curr_git_hash
        
        updated_config_file = self.logs_dir / 'config.yaml'
        
        with updated_config_file.open(mode='w') as rc:
            yaml.dump(self.run_config, rc, default_flow_style=False)            
        
        self.tx_sum_writer = SummaryWriter(str(self.logs_dir))
        
        if not log_every_x_batches:
            log_every_x_batches = batches_per_epoch
        self.game_logger_train = GameLogger(self.logs_dir, batches_per_epoch, 
                                      log_every_x_batches, name="train")
        self.game_logger_eval = GameLogger(self.logs_dir, eval_batches_per_epoch, 
                                      eval_log_every_x_batches, name="eval")
        
    @property    
    def epoch(self):
        return self._curr_epoch
    
    def load_logs(exp_dir: str) -> None:
        saved_yaml = Path(exp_dir) / 'logs' / 'config.yaml'
        with saved_yaml.open() as reader:
            run_config = yaml.safe_load(reader)
            
        logs_dir = Path(exp_dir) / 'logs'
        episodes = {log_path.stem: load_log(log_path) for log_path in logs_dir.glob('*.elg')}
        
        return run_config, episodes
        
    def export_summary(self):
        """ Override tensorboardX method which flushes dict after writing """
        path = self.logs_dir / 'all_scalars.json'
        scalar_dict = self.tx_sum_writer.scalar_dict
        json.dump(scalar_dict, path.open(mode='w'))
    
    def close(self):
        self.tx_sum_writer.close()
        
    def end_epoch(self):
        self._curr_epoch += 1
        
        
        
    
        
        
        
        
    
    
    
    


