#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Iterator, Tuple, Mapping
import pandas as pd
from functools import partial
from collections import abc
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from textlabs_agent_gym.training_logger import TrainingLogger, BatchEpisode, Episode

linestyles = ['-', '--', '-.', ':']
colors= ['blue', 'green', 'orange', 'red']

def get_by_index(l, i):
    return l[(min(i, len(l)))]
    

def nested_dict_iter(nested):
    for key, value in nested.items():
        if isinstance(value, abc.Mapping):
            yield from nested_dict_iter(value)
        else:
            yield key, value
            
class LogsViewer:
    def __init__(self, exp_dir: str, load_episodes: bool = True):
        self.exp_dir = Path(exp_dir)
        run_config, episodes = TrainingLogger.load_logs(exp_dir)
        self.episodes = episodes if load_episodes else None
        self.episode_dict = defaultdict(partial(defaultdict,partial(defaultdict, (partial(defaultdict, partial(defaultdict, Episode))))))
        self.run_config = run_config
        summary_path = self.exp_dir / 'logs' / 'all_scalars.json'
        self.scalar_data = {}
        if summary_path.exists():
            raw_scalar_data = json.load(summary_path.open())
            series = [pd.Series(v) for k,v in raw_scalar_data.items()]
            self.scalar_data = pd.DataFrame(series).T
            renamed_columns = [s.split('/')[-1] for s in raw_scalar_data.keys()]
            self.scalar_data.columns = renamed_columns
    
    def all_single_eps(self) -> Iterator[Tuple[int, Episode]]:
        if not self.episode_dict:
            self.episodes_to_dict()
        return nested_dict_iter(self.episode_dict)
    
    def episodes_to_dict(self) -> None:
        for log_name, episodes in self.episodes.items():
            for batch_ep in episodes:
                for ep in batch_ep.to_episodes():
                    self.episode_dict[batch_ep.name][ep.epoch_num][ep.batch_num][ep.slot_num] = \
                    ep
                
    def downsample_episodes(self, keep_every_nth_batch: int = 10, 
                            once_per_epoch: bool = True,
                            write_mode: str = "p") -> List[BatchEpisode]:
        ds_episodes = []
        if not once_per_epoch:
            for i in range(0, len(self.episodes), keep_every_nth_batch):
                ds_episodes.append(self.episodes[i])
                
        else:
            seen = {}
            for batch_ep in self.episodes:
                if not batch_ep.epoch_num in seen:
                    ds_episodes.append(batch_ep)
                    seen.update({batch_ep.epoch_num: True})
        
        # overwrite existing log file
        if write_mode == "o":
            episodes_log = self.exp_dir / 'logs' / 'episodes.log'
            with episodes_log.open(mode='w') as writer:
                for batch_ep in ds_episodes:
                    writer.write("\n" + str(batch_ep))
                    
        # write shrunk version to new file
        elif write_mode == "n":
            episodes_log = self.exp_dir / 'logs' / 'episodes_shrunk.log'
            with episodes_log.open(mode='w') as writer:
                for batch_ep in ds_episodes:
                    writer.write("\n" + str(batch_ep))
        
        # Do nothing, just return shrunk epsode list 
        elif write_mode == "p":
            pass
            
        
        return ds_episodes

class RunsSummarizer:
    def __init__(self, exp_dirs: List[str], load_episodes: bool = False):
        self.exps = {}
        # summary statistics of runs
        self.summary_data = None
        # raw data per timestep
        self.raw_data = None
        for exp_dir in exp_dirs:
            try:
                exp_name = Path(exp_dir).name
                lv = LogsViewer(exp_dir, load_episodes)
                self.exps[exp_name] = lv
            except Exception as e:
                print("Failed to load logs from {}".format(exp_dir))
                print(e)
        items = [(name, lv.scalar_data) for name, lv in self.exps.items()]
        self.init_multi_df(items)
        self.calc_last_stats()
                
    def init_multi_df(self, items: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        names, dfs = zip(*items)
        assert(len(names) == len(dfs))
        indices_list = []
        for i, df in enumerate(dfs):
            indices_list += [(names[i], n) for n in df.columns]
        
        # create MultiIndex first by exp and then by series name
        temp = pd.concat(dfs, axis=1).T
        temp.index = [i for i in range(len(temp.index))]
        self.raw_data = temp
        index = pd.MultiIndex.from_tuples(indices_list)
        self.raw_data.index = index
        self.summary_data = pd.DataFrame(index=index)
        return self.raw_data
    
    def calc_last_stats(self, last_x: int = 10) -> None:
        last_avgs = []
        for i, row in self.raw_data.iterrows():
            flipped = list(map(list, zip(*row.dropna())))
            ts, times, values = flipped
            last_avgs.append(np.mean(values[-last_x:]))
        self.summary_data['last_{}_steps_avg'.format(last_x)] = last_avgs
     
    def plot_series(self, series_names: List[str], rolling: int = 1) -> None:
        idx = pd.IndexSlice
        ax = plt.subplot(1,1,1)
        all_res = [self.raw_data.loc[idx[:,series_name],idx[:]] for series_name in series_names]
        for j, res in enumerate(all_res):        
            for n, (i, row) in enumerate(res.iterrows()):
                flipped = list(map(list, zip(*row.dropna())))
                ts, times, values = flipped
                ax.plot(times, pd.Series(values).rolling(rolling).mean(), 
                        label=', '.join(row.name), linestyle=get_by_index(linestyles, j), \
                        color=get_by_index(colors, n))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        return ax
        

if __name__ == "__main__":
    exp_root = Path(__file__).parent.parent.parent / 'exps'
    exp_dirs = [  
                str(exp_root / 'lstm_dqn_rcp_3_l2'),
                str(exp_root / 'lstm_dqn_rcp_3_l1')] 
    runs_summer = RunsSummarizer(exp_dirs, load_episodes=False)
    runs_summer.plot_series(['reward', 'eval_reward'], rolling=10)
        
    