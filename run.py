import argparse
import logging
import numpy as np
import random
import os
import scipy
import scipy.stats
import json

from agents.zendo import LLMNaiveZendo, LLMScientistZendo
from data.zendo import get_zendo_rules_and_examples, get_adv_zendo_rules_and_examples, ZendoModerator, ZendoGame, ZendoConfig, AdvZendoConfig, AdvZendoConfigStacking
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

    # secret_rules, examples = get_zendo_rules_and_examples()
    # zendo_config = ZendoConfig
    # rule = secret_rules[-3] # CHOOSE
    # game = ZendoGame(examples[rule], ['no', 'yes'])
    # log.info(f"Secret rules = {rule}\n Example: \n {game.to_text()}")

    secret_rules, rule_names, rule_programs, examples, test_sets = get_adv_zendo_rules_and_examples(config.dataset.extra)
    zendo_config = AdvZendoConfig
    # for k in secret_rules:
    #     log.info(f"K {k}")
    # for k in ['zeta', 'phi', 'upsilon', 'iota', 'kappa', 'omega', 'mu', 'nu', 'xi']:
    if config.seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [config.seed]
    alls, trues, falses = [], [], []
    alls2, trues2, falses2 = [], [], []
    raws = []
    extra_posteriors = []

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
            if isinstance(config.task_number, int):
                task_list = [task_list[config.task_number]]
            else:
                task_list = [task_list[x] for x in config.task_number]
        for k in task_list:

            set_seed(seed)
            config.seed = seed

            log.info(f"Running rule {k}")
            rule = secret_rules[k]
            game = ZendoGame(examples[k], ['yes'])
            log.info(f"Secret rules = {rule}\n Example: \n {game.to_text()}")
            stacking_flag = False
            for block in game.structures[0].blocks:
                if block.stacking is not None:
                    stacking_flag = True
                    zendo_config = AdvZendoConfigStacking

            moderator = ZendoModerator(rule, rule_programs[k])
            if config.agent.method == 'scientist':
                model = LLMScientistZendo(config, zendo_config, poor_llm=poor_llm, llm=llm, llm_exp=llm_exp, stacking_flag=stacking_flag)
                res, prob_res, support, phat_h_given_c = model.play_zendo(moderator, game, test_sets[k])
            elif config.agent.method == 'naive':
                model = LLMNaiveZendo(config, zendo_config, stacking_flag=stacking_flag)
                res, prob_res = model.play_zendo(moderator, game, test_sets[k])
                support, phat_h_given_c = [], []
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

        if config.dataset.extra:
            for h, phat in zip(support, phat_h_given_c):
                if h == 'The majority of blocks are red.':
                    extra_posteriors.append(phat)
            log.info(f'Final posterior: {dict(zip(support, phat_h_given_c))}')
                    

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

    posterior_m, posterior_h = mean_confidence_interval(extra_posteriors)
    log.info(f"Approximated posterior for 'The majority of blocks are red.' is {posterior_m} +/- {posterior_h}")

    log.info('Saving npy')
    with open(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'raw.npy'), 'wb') as f:
        np.save(f, np.asarray(raws))

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