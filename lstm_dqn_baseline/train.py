import numpy as np
import argparse
from pathlib import Path
import yaml
import gym

import textlabs_agent_gym  # Register all textworld environments.
from textlabs_agent_gym.envs import parse_env_id
from textlabs_agent_gym.training_logger import TrainingLogger
from tw_textlabs import EnvInfos
from eval_agent import eval_agent

from baseline_dqn_agent import BaselineDQNAgent


# List of additional information available during evaluation.
AVAILABLE_INFORMATION = EnvInfos(
    description=True, inventory=True,
    max_score=True, objective=True, entities=True, verbs=True,
    command_templates=True, admissible_commands=True,
    has_won=True, has_lost=True, intermediate_reward=True, facts=True
)


def _validate_requested_infos(infos: EnvInfos):
    msg = "The following information cannot be requested: {}"
    for key in infos.basics:
        if not getattr(AVAILABLE_INFORMATION, key):
            raise ValueError(msg.format(key))

    for key in infos.extras:
        if key not in AVAILABLE_INFORMATION.extras:
            raise ValueError(msg.format(key))


def train(config_fpath):
    
    with config_fpath.open() as reader:
        config = yaml.safe_load(reader)
    
    nb_epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    
    exp_dir = (Path(config['logging']['experiment_dir']) / 
                    config['checkpoint']['experiment_tag'])
    exp_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = str(config_fpath.parent)
    
    eval_freq = config['eval']['eval_freq']
    do_eval = eval_freq > 0
    
    print("Setting up training environment...")
    env_id = textlabs_agent_gym.make_batch(env_id=config['general']['env_id'],
                                      batch_size=batch_size,
                                      parallel=True)
    env = gym.make(env_id)
    env.seed(config['general']['random_seed'])
    env_dict = parse_env_id(config['general']['env_id'])
    
    
    n_batches = env_dict['n_games'] // batch_size
    config['general']['n_games'] = env_dict['n_games']
    
    eval_vocab = []
    eval_n_batches = 0
    if eval_freq:
        print("Setting up evaluation environment...")
        eval_batch_size = config['eval']['batch_size']
        eval_env_id = textlabs_agent_gym.make_batch(env_id=config['eval']['env_id'],
                                      batch_size=eval_batch_size,
                                      parallel=True)
        eval_env = gym.make(eval_env_id)
        eval_env.seed(config['general']['random_seed'])
        eval_env_dict = parse_env_id(config['eval']['env_id'])
        config['eval']['n_games'] = eval_env_dict['n_games']
        eval_n_batches = config['eval']['n_games'] // batch_size
        eval_vocab = eval_env.observation_space.vocab
        
    # assuming we have access to eval vocab at test time
    combined_vocab = env.observation_space.vocab + eval_vocab
    
    if config['vocab']['use_pregen_vocab']:
        vocab_path = config_fpath.parent.parent / config['vocab']['file']
        vocab = textlabs_agent_gym.envs.utils.load_vocab(vocab_path)
    else:
        vocab = combined_vocab
        
    agent = BaselineDQNAgent(word_vocab=vocab, config=config)
    agent.valid_verbs = config['vocab']['valid_actions']
    agent.valid_objs = config['vocab'].get('valid_objs', [])
    agent.valid_adjs = [agent.EOS_id]
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)
        
    
    train_logger = TrainingLogger(str(exp_dir), configs_dir,
                        batches_per_epoch=n_batches,
                        log_every_x_batches=config['logging']['batches_per_log'],
                        eval_batches_per_epoch=eval_n_batches,
                        eval_log_every_x_batches=config['eval']['batches_per_log'])
    
    game_logger = train_logger.game_logger_train
    summary = train_logger.tx_sum_writer
    print("Starting training...")
    for epoch_no in range(1, nb_epochs + 1):
        stats = {
            "scores": [],
            "intermediate_reward": [],
            "steps": []
        }
        for batch_no in range(n_batches):
            
            obs, infos = env.reset()
            agent.train()
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
            print("Epoch {}: Completed batch {}/{}. |"\
                  " {:2.1f} reward | {:2.1f} int. reward | {:4.1f} steps".format(epoch_no, 
                                                                 batch_no,
                                                                 n_batches,
                                                                 score,
                                                                 batch_int_reward,
                                                                 steps))
            game_logger.end_episode()
            

        
        score = sum(stats["scores"]) / env_dict['n_games']
        intermediate_reward = sum(stats["intermediate_reward"]) / env_dict['n_games']
        steps = sum(stats["steps"]) / env_dict['n_games']
        print("Epoch: {:3d} | {:2.1f} reward | {:2.1f} int. reward | {:4.1f}" \
              " steps".format(epoch_no, score, intermediate_reward, steps))
        
        # TensorboardX logging, need to use add_scalars according to 
        # https://github.com/lanpa/tensorboardX/issues/196
        if (epoch_no + 1) % config["training"]["scheduling"]["logging_frequency"] == 0:
            summary.add_scalars('data/scalar_group', {"reward": score,
                                             "interm_reward": intermediate_reward,
                                             "steps": steps }, 
                                                epoch_no + 1)
        train_logger.end_epoch()
        
        # Evaluation
        if do_eval and ((epoch_no + 1) % eval_freq == 0):
            eval_agent(config, eval_env, agent, train_logger)
            
        train_logger.export_summary()
            
    
    
    train_logger.export_summary()     
    train_logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument("config", metavar="path",
                       help="training configuration file.")
    args = parser.parse_args()
    config_file = Path(args.config)
    train(config_file)
