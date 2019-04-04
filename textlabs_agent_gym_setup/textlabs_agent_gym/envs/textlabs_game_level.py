#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join as pjoin
import numpy as np
from typing import Optional, Iterable, Tuple, Dict, Any
from pathlib import Path
import tw_textlabs
from tw_textlabs.gym.envs.textworld_games_env import TextworldGamesEnv
from tw_textlabs.envs.wrappers import Filter
from tw_textlabs.gym.spaces import text_spaces
from tw_textlabs.gym.envs.utils import shuffled_cycle
from tw_textlabs import LabGameOptions
from tw_textlabs.challenges.lab_game import make_game_from_level
from tw_textlabs.utils import encode_seeds

from textlabs_agent_gym.configs import generated_games_dir

DEFAULT_VOCAB_SIZE = 5
# TODO this should determine env uuid
MAX_GENERATION_ATTEMPTS = 15

class ExhaustedGameSearchLimit(RuntimeError):
    def __init__(self):
        msg = ("Failed to find enough valid games within allotted search limit \
               of %d attempts per requested game." % (MAX_GENERATION_ATTEMPTS))
        super().__init__(msg)

class TraningInitializationError(RuntimeError):
    def __init__(self, n_games, out_dir):
        msg = ("Failed to initialize. Check that all %d games were generated \
               properly in %s." % (n_games, out_dir))
        super().__init__(msg)
        
def extract_seed_from_game_name(game_name: str) -> int:
    return int(game_name.split('_')[-1])

class TextLabsGameLevel(TextworldGamesEnv):
    """
    Environment for generating and running TextLabs games.
    Need to call init_games() before generating games, and init_training()
    before training.
    """
    def __init__(self, level: int, n_games: int, generator_seed: int,
                 env_id: str, **kwargs):
        self.output_dir = generated_games_dir / str(env_id)
        self.games_collection = {}
        if not 'game_files' in kwargs:
            kwargs['game_files'] = self.collect_game_files(self.output_dir)
        super(TextLabsGameLevel, self).__init__(**kwargs)
        self.n_games = n_games
        self.level = level
        self.generator_seed = generator_seed
        self.seed(generator_seed)
        self.rng_games = np.random.RandomState(self.generator_seed + 1)
        self.rng_games_it = np.random.RandomState(self.generator_seed + 2)
        self.game_seeds = [self.rng_games.randint(np.iinfo(np.int32).max)
                                                for i in range(self.n_games)]
        self.seeds_iterator = shuffled_cycle(self.game_seeds, 
                                             rng=self.rng_games_it)
        
        
    
    def _skip_gen(self, nb_games: int = 1) -> None:
        """ Skip game generation seeds.

        Arguments:
            nb_games: Number of games to skip.
        """
        for _ in range(nb_games):
            next(self.seeds_iterator)
            
    
    def collect_game_files(self, gamefiles_dir: Path) -> Iterable[str]:
        for file_path in gamefiles_dir.glob('*.json'):
            stem = file_path.stem
            ulx_fpath = file_path.parent / (stem + '.ulx')
            if ulx_fpath.exists():
                seed = extract_seed_from_game_name(ulx_fpath.stem)
                self.games_collection[seed] = str(ulx_fpath)
        return list(self.games_collection.values())
        
    def _make_game(self, slot_seed: int, attempt_seed: int) -> str:
        options = LabGameOptions()
        options.seeds = attempt_seed
        game = make_game_from_level(self.level, options)
        hashid = encode_seeds([self.generator_seed, self.level] + [options.seeds[k] for k in sorted(options.seeds)])
        # slot seed in file name so we can reconstruct game collection
        # map upon loading an existing directory
        game_name = "{}_{}_{}".format(self.spec.id, hashid, slot_seed)
        options.path = str(self.output_dir / (game_name + ".ulx"))
        game_file = tw_textlabs.generator.compile_game(game, options)
        return game_file
    
    def _init_game_by_seed(self, slot_seed: int) -> Tuple[str, Dict[str, Any]]:
        if self.textworld_env is not None:
            self.textworld_env.close()

        current_gamefile = self.games_collection[slot_seed]
        env = tw_textlabs.start(current_gamefile)
        self.textworld_env = Filter(self.request_infos)(env)

        self.ob, infos = self.textworld_env.reset()
        infos['game_file'] = current_gamefile
        return self.ob, infos
        
        
    def _next_game(self):
        slot_seed = next(self.seeds_iterator)
        if not slot_seed in self.games_collection:
            self.games_collection[slot_seed] = self._attempt_make_game(slot_seed)
        if not self.games_collection[slot_seed] in self.gamefiles:
            self.gamefiles.append(self.games_collection[slot_seed])
        return self.games_collection[slot_seed]
    
    def _attempt_make_game(self, slot_seed: int,
                    max_attempts: int = MAX_GENERATION_ATTEMPTS) -> None:
        """ 
        Since we can't assure a game will be found for a given slot seed,
        we make up to `max_attempt` attempts.
        """
        found_game = False
        attempts = 0
        attempts_rng = np.random.RandomState(slot_seed)
        while not found_game:
            attempt_seed = attempts_rng.randint(np.iinfo(np.int32).max)
            # we will try max_attempts to make a game 
            # for each requested game
            try:
                # TODO check attempt_seed is saved
                game_file = self._make_game(slot_seed, attempt_seed)
                found_game = True
            except Exception as e:
                print(e)
                attempts += 1
                if attempts >= max_attempts:
                    raise ExhaustedGameSearchLimit()
        return game_file
        
        
    @property
    def initialized(self) -> bool:
        return ((len(self.gamefiles) == self.n_games) and
                (self.action_space.vocab_size > DEFAULT_VOCAB_SIZE) and
                (self.observation_space.vocab_size > DEFAULT_VOCAB_SIZE)
                )
    
    def init_games(self) -> None:
        """ Set output dir and collect already generated games. """
        self.output_dir = generated_games_dir / str(self.spec.id)
        self.collect_game_files()
        
    def init_training(self) -> None:
        """ 
        Iterate over all generated games and extract their vocabularies, to 
        initialize action/observation spaces.
        """
        games_iter = (tw_textlabs.Game.load(os.path.splitext(gamefile)[0] + ".json") for gamefile in self.gamefiles)
        vocab = tw_textlabs.text_utils.extract_vocab(games_iter)
        self.action_space = text_spaces.Word(max_length=8, vocab=vocab)
        self.observation_space = text_spaces.Word(max_length=200, vocab=vocab)
        self.seed(self.generator_seed)
        if not self.initialized:
            raise TraningInitializationError(self.n_games, 
                                             str(self.output_dir))
         
