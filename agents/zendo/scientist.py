import asyncio
import logging
import numpy as np
import random
import re
import pprint
from scipy.optimize import minimize, Bounds
from scipy.stats import norm

from data.zendo import ZendoGame, ZendoStructure
from prompts.zendo import *
from openai_hf_interface import create_llm
from utils import parse_listed_output, list_to_str
from ..utils import feedback_generator
from ..scientist import LLMScientist


log = logging.getLogger(__name__)


class LLMScientistZendo(LLMScientist):
    def __init__(self, config, zendo_config, poor_llm=None, llm=None, llm_exp=None):
        self.config = config
        self.zendo_config = zendo_config
        self.proposal_config = self.config.agent.proposal

        self.possible_ys = ['yes', 'no']
        self.rule_translation_prompt = rule_translation_prompt

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
            self.llm.set_default_kwargs({'timeout': 60})
        if self.llm_exp is None: 
            self.llm_exp = create_llm('gpt-4-1106-preview')
            self.llm_exp.setup_cache(cache_mode, database_path=config.database_path)
            self.llm_exp.set_default_kwargs({'timeout': 60})

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

    def play_zendo(self, moderator, game, test_set=None):
        done = False

        c = game

        while len(c) < self.config.n_gameplay:
            log.info(f'Iteration {len(c)}')
            # TODO: Track hypotheses, queried, answer, cost
            query, _ = asyncio.run(self.a_get_query_x(c))

            log.info(f'Query: \n{query.to_text()}')

            # log.info(f'Moderator verdict: ')
            # verdict = input()

            verdict = moderator.query(query)
            log.info(f'LLM moderator verdict: {verdict}')

            log.info([f'Cost at iteration {len(c)}', self.llm.get_info(), self.llm_exp.get_info(), self.poor_llm.get_info(), len(self.cache_h_program)])

            c = c.append_and_retnew(query, verdict)

        res = None
        prob_res = None
        if test_set is not None:
            res, prob_res = asyncio.run(self.a_eval_test_set(c, test_set))
        
        log.info(['Final cost', self.llm.get_info(), self.llm_exp.get_info(), self.poor_llm.get_info(), len(self.cache_h_program)])

        return res, prob_res
        
    def get_previous_c(self, c):
        previous_c = ZendoGame(c.xs[:-1], c.ys[:-1])
        return previous_c
    
    def get_sub_c(self, c):
        internal_ct = 0
        while internal_ct == 0 or (internal_ct < 20 and sub_c.to_text() in self.cache_sub_c):
            x1, y1, x2, y2 = c.samp_two()
            if x2 is None:
                sub_c = ZendoGame([x1], [y1])
            else:
                sub_c = ZendoGame([x1, x2], [y1, y2])
            internal_ct += 1
        return sub_c
    
    async def first_propose_hs(self, c):
        good_x, _ = c.samp_good_bad()
        sub_c = good_x

        potential_hs = await self.a_sample_potential_hs(sub_c)
        return potential_hs
    
    async def a_sample_potential_hs(self, sub_c):
        if sub_c.to_text() in self.cache_sub_c:
            return self.cache_sub_c[sub_c.to_text()]

        # See all attributes at once
        if self.proposal_config.name.startswith('is'):
            orders =[
                [0, 1, 2, 3, 4],
            ]
            outputs = await self.llm.aprompt([propose_h_all_no_neg_prompt.format(text_c=sub_c.to_text(order=order), 
                                                                          att_summary=self.zendo_config.get_spec().replace('\n', ', '), 
                                                                          num=self.proposal_config.num_hypotheses_per_call * 3) for order in orders], 
                                                                          temperature=0, seed=self.config.seed)
        elif self.proposal_config.name == 'particle_filter':
            outputs = await self.llm.aprompt([basic_propose_h_prompt.format(x=sub_c.to_text(att=att), 
                                                                            att=att, 
                                                                            att_choices=self.zendo_config.att_choices[att], 
                                                                            num=self.proposal_config.num_hypotheses_per_call) 
                                                                            for att in self.zendo_config.att if att != 'touchings'], temperature=0, seed=self.config.seed)
            # outputs = []
            # touching_outputs = ['\n'.join([f'{idx}. There is at least one block' for idx in range(1, 6)])]
            # TODO What do you want to do here?
            touching_outputs = ['\n'.join([f'{idx}. There is at least one block' for idx in range(1, 2)])]
            outputs = outputs + touching_outputs
        else:
            raise NotImplementedError
        
        potential_hs = np.concatenate([parse_listed_output(output) for output in outputs])
        potential_hs = np.random.permutation(potential_hs)
        self.cache_sub_c[sub_c.to_text()] = potential_hs
        return self.cache_sub_c[sub_c.to_text()]

    async def a_get_rejuvenation_options(self, h, x, y, c):
        outputs = await self.llm.aprompt([new_evolve_h_prompt.format(h=h, text_y='' if y == 'yes' else 'NOT', 
                                                                     x=x.to_text(),
                                                                     num=self.proposal_config.num_hypotheses_per_call)], temperature=0, seed=self.config.seed)
        rules = sum([parse_listed_output(output) for output in outputs], [])
        rules = np.asarray([rule.split('->')[-1].strip(" '\n") for rule in rules])
        probs = await self.a_score_joints(rules, c)
        probs = np.asarray(probs, dtype=np.float64)
        return rules, probs

    async def a_refine_hs(self, hs, c):
        count_score_grid = await self.a_count_score_hs(hs, c)
        count_score_grid = np.asarray(count_score_grid)
        count_score_sums = np.sum(count_score_grid, -1)

        best_idx = np.argmax(count_score_sums)
        best_h = hs[best_idx]

        structures, labels = np.asarray(c.structures), np.asarray(c.labels)
        mistake_tf_indices = (count_score_grid[best_idx] == 0)
        mistake_structures, mistake_labels = structures[mistake_tf_indices], labels[mistake_tf_indices]

        if len(mistake_structures) > 2:
            fb_indices = np.random.choice(len(mistake_structures), 2, replace=False)
            fb_structures, fb_labels = mistake_structures[fb_indices], mistake_labels[fb_indices]
        else:
            fb_structures, fb_labels = mistake_structures, mistake_labels

        feedback = feedback_generator(fb_structures, fb_labels, self.possible_ys)

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
                structure = ZendoStructure(txt=xs_txt)
                if len(structure) > 0:
                    res.append(ZendoStructure(txt=xs_txt))
                if len(res) == self.config.agent.num_xs:
                    break
        elif self.config.agent.active_method == 'random':
            res = [ZendoStructure(random_block_rng=np.random.default_rng(np.random.randint(500))) for _ in range(self.config.agent.num_xs)]
            # for x in res:
            #     log.info(x.to_text())
        elif self.config.agent.active_method == 'llm':
            res = await self.a_sample_proposal_r_llm(hs)
            res = [ZendoStructure(txt=res)]
        else:
            raise NotImplementedError
        return res

    async def a_sample_proposal_r_info_gain(self, h, rng=None):
        if h in self.cache_h:
            return self.cache_h[h]
        
        outputs = await self.llm_exp.aprompt([propose_x_prompt.format(h=h, 
                                                                    spec=self.zendo_config.get_spec(rng), 
                                                                    example_block=self.zendo_config.example_block)], temperature=0, seed=self.config.seed)

        xs = []
        cur_output = outputs[0]
        i = 1
        lst = ['(conforms with the rule) Structure 1:\n', '(violates the rule) Structure 2:\n', 'ljgsdf']
        while len(cur_output.strip(' .\n')) > 0:
            item = cur_output.split(lst[i], 1)[0]
            xs.append(item.strip(' .\n')[len(lst[i-1]):] + '\n')
            cur_output = cur_output[len(item):]
            i += 1
        self.cache_h[h] = xs
        return self.cache_h[h]
    
    async def a_sample_proposal_r_llm(self, hs):
        if list_to_str(hs) in self.cache_hs:
            return self.cache_hs[list_to_str(hs)]
        
        outputs = await self.llm_exp.aprompt([propose_llm_x_prompt.format(hs=list_to_str(hs), 
                                                                          spec=self.zendo_config.get_spec(), 
                                                                          example_block=self.zendo_config.example_block)], temperature=0, seed=self.config.seed)

        cur_output = outputs[0]
        x = cur_output.split('Structure 1:', 1)[1].split('End of structure 1', 1)[0] + '\n'

        self.cache_hs[list_to_str(hs)] = x
        return self.cache_hs[list_to_str(hs)]
    
    async def a_eval_test_set(self, c, test_set):
        _ = await self.a_sample_proposal_q(c)
        predicted_y_dists = await asyncio.gather(*[self.a_dist_y_given_cx(c, x) for x in test_set])
        log.info(f'predicted y dist {predicted_y_dists}')
        return [bool(y_dist[0] > y_dist[1]) for y_dist in predicted_y_dists], [y_dist[0] for y_dist in predicted_y_dists]