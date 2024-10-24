import argparse
import logging
import numpy as np
import random
import os
import scipy
import scipy.stats
import json

from agents.zendo import LLMNaiveZendo, LLMScientistZendo
from agents.acre import LLMNaiveACRE, LLMScientistACRE
from data.zendo import get_zendo_rules_and_examples, get_adv_zendo_rules_and_examples, ZendoModerator, ZendoGame, ZendoConfig, AdvZendoConfig
from data.acre import get_acre_rules_and_examples, ACREModerator, ACREGame
from openai_hf_interface import create_llm

import hydra
from omegaconf import OmegaConf
from pathlib import Path


log = logging.getLogger(__name__)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.asarray(data)
    n = len(a)
    m, se = np.mean(a, 0), scipy.stats.sem(a, 0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    # return m, h
    return m, se


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    # Path(config.save_folder).mkdir(parents=True, exist_ok=True)
    # log_to_file(os.path.join(config.save_folder, 'run.log'))
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    if config.dataset.name.startswith('zendo'):
        # secret_rules, examples = get_zendo_rules_and_examples()
        # zendo_config = ZendoConfig
        # rule = secret_rules[-3] # CHOOSE
        # game = ZendoGame(examples[rule], ['no', 'yes'])
        # log.info(f"Secret rules = {rule}\n Example: \n {game.to_text()}")

        secret_rules, rule_names, rule_programs, examples, test_sets = get_adv_zendo_rules_and_examples(config.dataset.extra)
        zendo_config = AdvZendoConfig
        if config.seed == -1:
            seeds = [0, 1, 2, 3, 4]
        else:
            seeds = [config.seed]
        alls, trues, falses = [], [], []
        alls2, trues2, falses2 = [], [], []
        raws = []

        # ---- Construct llms ----
        log.info(f'Creating llms [using memory: {config.use_memory}]...')
        cache_mode = 'disk_to_memory' if config.use_memory else 'disk'

        poor_llm = create_llm('gpt-3.5-turbo')
        poor_llm.setup_cache(cache_mode, database_path=config.database_path)

        llm = create_llm('gpt-4-1106-preview')
        llm.setup_cache(cache_mode, database_path=config.database_path)
        llm.set_default_kwargs({'timeout': 60, 'request_timeout': 60})

        llm_exp = create_llm('gpt-4-1106-preview')
        llm_exp.setup_cache(cache_mode, database_path=config.database_path)
        llm_exp.set_default_kwargs({'timeout': 60, 'request_timeout': 60})
        log.info(f'Creating llms [using memory: {config.use_memory}]... DONE')
        # -------------------------

        for seed in seeds:
            log.info(f'SEED {seed}')
            all_scores = []
            true_scores = []
            false_scores = []
            all_prob_scores = []
            true_prob_scores = []
            false_prob_scores = []
            raw_scores = []
            task_list = rule_names
            # task_list = ['zeta', 'phi', 'upsilon', 'iota', 'kappa', 'omega', 'mu', 'nu', 'xi']
            if config.task_number != -1:
                task_list = [task_list[config.task_number]]
            for k in task_list:

                set_seed(seed)
                config.seed = seed

                log.info(f"Running rule {k}")
                rule = secret_rules[k]
                game = ZendoGame(examples[k], ['yes'])
                log.info(f"Secret rules = {rule}\n Example: \n {game.to_text()}")

                if config.agent.method == 'scientist':
                    model = LLMScientistZendo(config, zendo_config, poor_llm=poor_llm, llm=llm, llm_exp=llm_exp)
                elif config.agent.method == 'naive':
                    model = LLMNaiveZendo(config, zendo_config)

                moderator = ZendoModerator(rule, rule_programs[k])
                res, prob_res = model.play_zendo(moderator, game, test_sets[k])
                log.info(f"Rule {k}: res {res}")
                log.info(f"Rule {k}: prob res {prob_res}")
                true_res = ([True] * 4) + ([False] * 4)

                res = np.asarray(res)
                true_res = np.asarray(true_res)
                log.info(f"Rule {k}: score {np.sum(res == true_res)}")
                all_scores.append(np.sum(res == true_res))
                true_scores.append(np.sum(res[:4] == true_res[:4]))
                false_scores.append(np.sum(res[4:] == true_res[4:]))

                if prob_res is not None:
                    prob_res = np.asarray(prob_res)
                    log.info(f"Rule {k}: prob score {np.sum(prob_res[:4] + (1 - prob_res[4:]))}")
                    all_prob_scores.append(np.sum(prob_res[:4] + (1 - prob_res[4:])))
                    true_prob_scores.append(np.sum(prob_res[:4]))
                    false_prob_scores.append(np.sum(1 - prob_res[4:]))

                    raw_scores.append(prob_res[:4])
                    raw_scores.append(1 - prob_res[4:])
            log.info(f'all_scores: {all_scores}')
            log.info(f'true_scores: {true_scores}')
            log.info(f'false_scores: {false_scores}')
            alls.append(all_scores)
            trues.append(true_scores)
            falses.append(false_scores)

            log.info(f'all_prob_scores: {all_prob_scores}')
            log.info(f'true_prob_scores: {true_prob_scores}')
            log.info(f'false_prob_scores: {false_prob_scores}')
            log.info(f'avg_prob_scores: {np.mean(all_prob_scores)}')
            alls2.append(all_prob_scores)
            trues2.append(true_prob_scores)
            falses2.append(false_prob_scores)

            raws.append(np.concatenate(raw_scores))

        all_m, all_h = mean_confidence_interval(alls)
        true_m, true_h = mean_confidence_interval(trues)
        false_m, false_h = mean_confidence_interval(falses)

        log.info(f'avg all_scores: {list(all_m)} +/- {list(all_h)}')
        log.info(f'avg true_scores: {list(true_m)} +/- {list(true_h)}')
        log.info(f'avg false_scores: {list(false_m)} +/- {list(false_h)}')

        all_m2, all_h2 = mean_confidence_interval(alls2)
        true_m2, true_h2 = mean_confidence_interval(trues2)
        false_m2, false_h2 = mean_confidence_interval(falses2)

        log.info(f'avg all_prob_scores: {list(all_m2)} +/- {list(all_h2)}')
        log.info(f'avg true_prob_scores: {list(true_m2)} +/- {list(true_h2)}')
        log.info(f'avg false_prob_scores: {list(false_m2)} +/- {list(false_h2)}')
        log.info(f'avg avg_prob_scores: {np.mean(all_m2)}')

        log.info(f'Dumping cache to disk...')
        poor_llm.cache.dump_to_disk()
        llm.cache.dump_to_disk()
        llm_exp.cache.dump_to_disk()
        log.info(f'Dumping cache to disk... DONE')
    else:
        rules, train_games, test_games = get_acre_rules_and_examples()

        # TODO
        rules = rules[:20]
        train_games = train_games[:20]
        test_games = test_games[:20]

        log.debug(train_games[0].to_text())

        # ---- Construct llms ----
        log.info(f'Creating llms [using memory: {config.use_memory}]...')
        cache_mode = 'disk_to_memory' if config.use_memory else 'disk'

        poor_llm = create_llm('gpt-3.5-turbo')
        poor_llm.setup_cache(cache_mode, database_path=config.database_path)

        llm = create_llm('gpt-4-1106-preview')
        llm.setup_cache(cache_mode, database_path=config.database_path)
        llm.set_default_kwargs({'timeout': 60, 'request_timeout': 60})

        llm_exp = create_llm('gpt-4-1106-preview')
        llm_exp.setup_cache(cache_mode, database_path=config.database_path)
        llm_exp.set_default_kwargs({'timeout': 60, 'request_timeout': 60})
        log.info(f'Creating llms [using memory: {config.use_memory}]... DONE')
        # -------------------------

        alls = []
        alls2 = []
        all_prob_res = []
        raws = []
        alls3 = []
        alls4 = []
        alls5 = []
        for idx, (rule, train_game, test_game) in enumerate(zip(rules, train_games, test_games)):
            if idx < 13: 
                continue
            if config.seed == -1:
                set_seed(0)
            else:
                set_seed(config.seed)
            log.info(f'Game {idx}')
            log.info(f'Game {idx} Rule = {rule}')
            log.info(f'Game {idx} Train game = {train_game.to_text()}')
            if config.agent.method == 'scientist':
                model = LLMScientistACRE(config, list(rule.keys()), poor_llm=poor_llm, llm=llm, llm_exp=llm_exp)
            elif config.agent.method == 'naive':
                model = LLMNaiveACRE(config, list(rule.keys()), llm=llm)
            moderator = ACREModerator(rule)
            res, prob_res, roc_auc, f1, task_solved = model.play(moderator, train_game, test_game)

            all_prob_res.append(prob_res)
            alls.append(np.mean(res))
            alls2.append(np.mean(prob_res))
            alls3.append(roc_auc)
            alls4.append(f1)
            alls5.append(task_solved)

            log.info(f'Game {idx} Ground Truth Rule = {rule}')
            # log.info(f'Game {idx} Test game = {test_game.to_text()}')

            log.info(f'Game {idx}: res {res}')
            log.info(f"Game {idx}: prob res {prob_res}")
            log.info(f'Game {idx}: score {np.mean(res)}')
            log.info(f"Game {idx}: prob score {np.mean(prob_res)}")
            log.info(f"Game {idx}: roc auc {roc_auc}")
            log.info(f"Game {idx}: f1 {f1}")
            log.info(f"Game {idx}: task_solved {task_solved}")

            log.info(f'Running avg score = {np.mean(alls2)}')
            log.info(f'Running avg prob score = {np.mean(alls2)}')
            log.info(f'Running avg roc auc = {np.mean(alls3)}')
            log.info(f'Running avg f1 = {np.mean(alls4)}')
            log.info(f'Running avg task_solved = {np.mean(alls5)}')

        raws = np.concatenate(all_prob_res)

        all_m, all_h = mean_confidence_interval(alls)
        log.info(f'avg avg_score: {all_m} +/- {all_h}')

        all2_m, all2_h = mean_confidence_interval(alls2)
        log.info(f'avg avg_prob_scores: {all2_m} +/- {all2_h}')

        all3_m, all3_h = mean_confidence_interval(alls3)
        log.info(f'avg roc auc: {all3_m} +/- {all3_h}')

        all4_m, all4_h = mean_confidence_interval(alls4)
        log.info(f'avg f1: {all4_m} +/- {all4_h}')

        all5_m, all5_h = mean_confidence_interval(alls5)
        log.info(f'avg task_solved: {all5_m} +/- {all5_h}')

        log.info(f'Dumping cache to disk...')
        try:
            poor_llm.cache.dump_to_disk()
            llm.cache.dump_to_disk()
            llm_exp.cache.dump_to_disk()
            log.info(f'Dumping cache to disk... DONE')
        except:
            log.info(f'No dumping')


    with open(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'raw.npy'), 'wb') as f:
        np.save(f, np.asarray(raws))




if __name__ == '__main__':
    main()