#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from pathlib import Path
from tw_textlabs import EnvInfos


MAX_NB_STEPS_PER_EPISODE = 50

# List of additional information available during evaluation.
AVAILABLE_EVAL_INFORMATION = EnvInfos(
    description=True, inventory=True,
    max_score=True, objective=True, entities=True, verbs=True,
    command_templates=True, admissible_commands=True,
    has_won=True, has_lost=True, intermediate_reward=True, facts=True
)


def _validate_requested_infos(infos: EnvInfos):
    msg = "The following information cannot be requested: {}"
    for key in infos.basics:
        if not getattr(AVAILABLE_EVAL_INFORMATION, key):
            raise ValueError(msg.format(key))

    for key in infos.extras:
        if key not in AVAILABLE_EVAL_INFORMATION.extras:
            raise ValueError(msg.format(key))


def eval_agent(config, env, agent, train_logger):
    
    agent.eval()
    
    batch_size = config['eval']['batch_size']
    n_games = config['eval']['n_games']
    n_batches = config['eval']['n_games'] // batch_size
    
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)
    
    
    summary = train_logger.tx_sum_writer
    game_logger = train_logger.game_logger_eval
    
    print("Starting evaluation...")

    stats = {
        "scores": [],
        "intermediate_reward": [],
        "steps": []
    }
    for batch_no in range(n_batches):
        
        obs, infos = env.reset()
        game_files = infos['game_file'] if 'game_file' in infos else []
        game_logger.reset(objectives=infos['objective'], 
                          game_files=game_files)
        scores = [0] * len(obs)
        dones = [False] * len(obs)
        steps = [0] * len(obs)
        batch_stats = {
            "intermediate_reward": np.array([0] * len(obs))
        }
        while not all(dones):
            # Increase step counts.
            steps = [step + int(not done) for step, done in zip(steps, dones)]
            commands = agent.act(obs, scores, dones, infos)
            obs, scores, dones, infos = env.step(commands)
            # mask[i] == 1 iff game[i] not done, 0 o.w
            mask = np.logical_not(np.array(dones))
            batch_stats["intermediate_reward"] += \
                                    np.multiply(
                                    np.array(infos["intermediate_reward"])
                                    , mask)
            game_logger.log_step(commands, obs, scores, 
                                 infos["intermediate_reward"])
            

        # Let the agent knows the game is done.
        agent.act(obs, scores, dones, infos)
        stats["scores"].extend(scores)
        stats["steps"].extend(steps)
        stats["intermediate_reward"].extend(batch_stats["intermediate_reward"].tolist())
        
        
        score = sum(scores) / batch_size
        steps = sum(steps) / batch_size
        batch_int_reward = np.mean(batch_stats["intermediate_reward"])
        print("Evaluation: Completed batch {}/{}. |"\
              " {:2.1f} reward | {:2.1f} int. reward | {:4.1f} steps".format( 
                                                             batch_no,
                                                             n_batches,
                                                             score,
                                                             batch_int_reward,
                                                             steps))
        game_logger.end_episode()
        

    
    score = sum(stats["scores"]) / n_games
    intermediate_reward = sum(stats["intermediate_reward"]) / n_games
    steps = sum(stats["steps"]) / n_games
    print("Evaluation complete: {:2.1f} reward | {:2.1f} int. reward | {:4.1f}" \
          " steps".format(score, intermediate_reward, steps))
    
    # TensorboardX logging, need to use add_scalars according to 
    # https://github.com/lanpa/tensorboardX/issues/196
    summary.add_scalars('data/scalar_group', {"eval_reward": score,
                                     "eval_interm_reward": intermediate_reward,
                                     "eval_steps": steps }, 
                                        train_logger.epoch + 1)