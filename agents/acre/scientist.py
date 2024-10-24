import asyncio
import logging
import numpy as np
from scipy.stats import norm

from data.acre import ACREGroup, ACREGame
from prompts.acre import *
from openai_hf_interface import create_llm
from utils import parse_listed_output
from ..scientist import LLMScientist
from sklearn.metrics import roc_auc_score, f1_score
from ..utils import feedback_generator

log = logging.getLogger(__name__)
# log.level = logging.DEBUG


class LLMScientistACRE(LLMScientist):
    def __init__(self, config, all_objects, poor_llm=None, llm=None, llm_exp=None):
        self.config = config
        self.proposal_config = self.config.agent.proposal

        self.possible_ys = ['on', 'off']
        self.rule_translation_prompt = rule_translation_prompt
        self.all_objects = all_objects

        self.basic_propose_h_prompt = basic_propose_h_prompt
        if self.proposal_config.name.startswith('is') and self.proposal_config.prompt_w_example:
            self.basic_propose_h_prompt = basic_propose_h_prompt_w_example

        self.first_propose = True

        self.poor_llm = poor_llm
        self.llm = llm
        self.llm_exp = llm_exp

        cache_mode = 'disk_to_memory' if self.config.use_memory else 'disk'
        if self.poor_llm is None: 
            self.poor_llm = create_llm('gpt-3.5-turbo')
            self.poor_llm.setup_cache(cache_mode, database_path=config.database_path)
        if self.llm is None: 
            self.llm = create_llm('gpt-4-1106-preview')
            self.llm.setup_cache(cache_mode, database_path=config.database_path)
            self.llm.set_default_kwargs({'timeout': 60, 'request_timeout': 60})
        if self.llm_exp is None: 
            self.llm_exp = create_llm('gpt-4-1106-preview')
            self.llm_exp.setup_cache(cache_mode, database_path=config.database_path)
            self.llm_exp.set_default_kwargs({'timeout': 60, 'request_timeout': 60})

        self.particles = []
        self.particle_weights = []
        self.pf_checkpoint = 0

        self.cache_h_program = {}
        self.cache_c = {}
        self.cache_h = {}
        self.cache_cx = {}
        self.cache_sub_c = {}
        self.cache_prior_h = {}
        self.cache_hs = {}

        if self.proposal_config.deterministic:
            self.thetas = np.ones(1)
            self.theta_priors = np.ones(1)
            self.deltas = np.ones(1)
            self.delta_priors = np.ones(1)
        else:
            self.thetas = np.arange(0.5, 1.01, 0.01)
            self.theta_priors = np.exp(np.asarray([norm.logpdf(theta, self.config.agent.theta_mean, 0.1) for theta in self.thetas]))
            self.deltas = np.arange(0.5, 1.01, 0.01)
            self.delta_priors = np.exp(np.asarray([norm.logpdf(delta, self.config.agent.delta_mean, 0.01) for delta in self.deltas]))
        self.all_priors = np.tile(self.theta_priors, len(self.deltas)) * np.repeat(self.delta_priors, len(self.thetas))

    def play(self, moderator, game, test_game):
        c = game

        # # Do it on the first three first
        # for i in range(-3, 0):
        #     log.info(f'Iteration {len(c) + i}, No active learning')
        #     log.info(f"Seen game: {ACREGame([c.xs[i-1]], [c.ys[i-1]]).to_text()}")
        #     _ = asyncio.run(self.a_sample_proposal_q(ACREGame(c.xs[:i], c.ys[:i])))

        # number_of_rounds = len(c.xs[-1]) # Let number of rounds equal to number of objects
        number_of_rounds = 7

        log.info(f'Num rounds {number_of_rounds}')

        ct = 0
        log.info(f"Seed game: {ACREGame([c.xs[-1]], [c.ys[-1]]).to_text()}")
        while ct < number_of_rounds:
            log.info(f'Iteration {len(c)}')
            query, _ = asyncio.run(self.a_get_query_x(c))

            log.info(f'Query: \n{query.to_text()}')

            verdict = moderator.query(query)
            log.info(f'Moderator verdict: {verdict}')

            log.info([f'Cost at iteration {len(c)}', self.llm.get_info(), self.llm_exp.get_info(), self.poor_llm.get_info(), len(self.cache_h_program)])

            c = c.append_and_retnew(query, verdict)
            ct += 1

        res = None
        if test_game is not None:
            res = asyncio.run(self.a_eval_test_set(c, test_game))
        
        log.info(['Final cost', self.llm.get_info(), self.llm_exp.get_info(), self.poor_llm.get_info(), len(self.cache_h_program)])
        return res
    
    async def a_score_h(self, h):
        return 1
    
    def get_previous_c(self, c):
        previous_c = ACREGame(c.xs[:-1], c.ys[:-1])
        return previous_c
    
    def get_sub_c(self, c):
        internal_ct = 0
        sub_c = c
        while internal_ct == 0 or (internal_ct < 20 and sub_c.to_text() in self.cache_sub_c):
            sub_c = c.permute(rng=np.random.default_rng(np.random.randint(500)))
            internal_ct += 1
        return sub_c
    
    async def first_propose_hs(self, c):
        potential_hs = await self.a_sample_potential_hs(c)
        return potential_hs
    
    async def a_sample_potential_hs(self, sub_c):
        if sub_c.to_text() in self.cache_sub_c:
            return self.cache_sub_c[sub_c.to_text()]

        # See all attributes at once
        outputs = await self.llm.aprompt([self.basic_propose_h_prompt.format(text_c=sub_c.to_text(),
                                                                        num=self.proposal_config.num_hypotheses_per_call * 3)], temperature=0, seed=self.config.seed)
        
        potential_hs = np.concatenate([parse_listed_output(output) for output in outputs])
        # potential_hs = np.random.permutation(potential_hs) # Removing this to ensure no randomness when set seed
        self.cache_sub_c[sub_c.to_text()] = potential_hs
        log.debug(f'Prompt:\n: {self.basic_propose_h_prompt.format(text_c=sub_c.to_text(), num=self.proposal_config.num_hypotheses_per_call * 3)}\nOutput:\n{outputs}')
        return self.cache_sub_c[sub_c.to_text()]
    
    def strip_prefix(self, rule):
        for rule_mod_type in ['Additional conjunction: ', 'Additional attribute: ', 'Attribute change: ']:
            rule = rule.split(rule_mod_type)[-1]
        return rule

    async def a_get_rejuvenation_options(self, h, x, y, c):
        outputs = await self.llm.aprompt([new_evolve_h_prompt.format(h=h, text_y='' if y == 'on' else 'NOT', 
                                                                     x=x.to_text(),
                                                                     num=self.proposal_config.num_hypotheses_per_call)], temperature=0, seed=self.config.seed)
        rules = sum([parse_listed_output(output) for output in outputs], [])
        rules = np.asarray([self.strip_prefix(rule.split('->')[-1]).strip(" '\n") for rule in rules])
        probs = await self.a_score_joints(rules, c)
        probs = np.asarray(probs, dtype=np.float64)
        log.debug(['REJU', h, rules, probs])
        return rules, probs

    async def a_refine_hs(self, hs, c):
        count_score_grid = await self.a_count_score_hs(hs, c)
        count_score_grid = np.asarray(count_score_grid)
        count_score_sums = np.sum(count_score_grid, -1)

        best_idx = np.argmax(count_score_sums)
        best_h = hs[best_idx]

        xs, ys = np.asarray(c.xs), np.asarray(c.ys)
        mistake_tf_indices = (count_score_grid[best_idx] == 0)
        mistake_xs, mistake_ys = xs[mistake_tf_indices], ys[mistake_tf_indices]

        fb_xs, fb_ys = mistake_xs, mistake_ys

        feedback = feedback_generator(fb_xs, fb_ys, self.possible_ys)

        outputs = await self.llm.aprompt([refine_prompt.format(h=best_h, feedback=feedback, num=self.proposal_config.num_hypotheses_per_call * 3)], temperature=0, seed=self.config.seed)
        rules = sum([parse_listed_output(output) for output in outputs], [])
        return rules
    
    async def a_sample_proposal_r_handler(self, hs):
        if self.config.agent.active_method == 'info_gain':
            all_xs_txt = await asyncio.gather(*[self.a_sample_proposal_r_info_gain(h, rng=np.random.default_rng(np.random.randint(500))) for h in hs])
            all_xs_txt = np.concatenate(all_xs_txt)
            all_xs_txt = np.random.permutation(all_xs_txt)
            res = []
            for xs_txt in all_xs_txt:
                structure = ACREGroup(txt=xs_txt, constraints=self.all_objects)
                if len(structure) > 0:
                    res.append(structure)
                if len(res) == self.config.agent.num_xs:
                    break
        elif self.config.agent.active_method == 'random':
            raise NotImplementedError
        elif self.config.agent.active_method == 'llm':
            raise NotImplementedError
        else:
            raise NotImplementedError
        log.debug(f'SAMPLE R\n:{res}')
        return res

    async def a_sample_proposal_r_info_gain(self, h, rng=None):
        if h in self.cache_h:
            return self.cache_h[h]
        outputs = await self.llm_exp.aprompt([propose_x_prompt.format(h=h, 
                                                                      all_objects=', '.join(self.all_objects))], temperature=0, seed=self.config.seed)
        xs = []
        cur_output = outputs[0]
        i = 1
        lst = ['light on group of objects: ', 'light off group of objects: ', 'ljgsdf']
        while len(cur_output.strip(' .\n')) > 0:
            item = cur_output.split(lst[i], 1)[0]
            xs.append(item.strip(' .\n')[len(lst[i-1]):])
            cur_output = cur_output[len(item):]
            i += 1
        log.debug(['OUTPUTS', outputs])
        log.debug(['XS', xs])
        self.cache_h[h] = xs
        return self.cache_h[h]

    async def a_eval_test_set(self, c, test_game):
        _ = await self.a_sample_proposal_q(c)
        predicted_y_dists = await asyncio.gather(*[self.a_dist_y_given_cx(c, x) for x in test_game.xs])
        correct_indices = [0 if y == 'on' else 1 for y in test_game.ys]
        roc_auc = roc_auc_score(correct_indices, [y_dist[1] for y_dist in predicted_y_dists])
        f1 = f1_score(correct_indices, [0 if y_dist[0] >= y_dist[1] else 1 for y_dist in predicted_y_dists], pos_label=0)
        task_solved = bool(f1 == 1.0)
        return [bool(y_dist[correct_idx] > y_dist[correct_idx ^ 1]) for y_dist, correct_idx in zip(predicted_y_dists, correct_indices)], \
            [y_dist[correct_idx] for y_dist, correct_idx in zip(predicted_y_dists, correct_indices)], roc_auc, f1, task_solved
        