import asyncio
import logging
import numpy as np
import random
import re
import pprint
from scipy.optimize import minimize, Bounds
from scipy.stats import norm

from data.zendo import ZendoGame, ZendoBlock, ZendoStructure
from prompts.zendo import *
from openai_hf_interface import create_llm
from utils import parse_listed_output, list_to_str
from .utils import systematic_resample, feedback_generator, resample_optimal


log = logging.getLogger(__name__)


class LLMScientistZendo:
    def __init__(self, config, zendo_config, poor_llm=None, llm=None, llm_exp=None, stacking_flag=False):
        self.config = config
        self.zendo_config = zendo_config
        self.proposal_config = self.config.agent.proposal

        self.stacking_flag = stacking_flag
        
        if stacking_flag:
            self.att_par = """color (blue/red/green) 
size (small/medium/large)
orientation (upright/left/right/strange)
groundedness (grounded/ungrounded/stacking),
touching (which other blocks they do and do not touch).
"""
            self.groundedness_param_msg = ':param groundedness_str: str (grounded/ungrounded/stacking)'
            self.stacking_note = "For 'stacking' block, please indicate which block it is stacking on by appending (stacking) to one of the blocks it touches"
        else:
            self.att_par = """color (blue/red/green) 
size (small/medium/large)
orientation (upright/left/right/strange)
groundedness (grounded/ungrounded),
touching (which other blocks they do and do not touch)."""
            self.groundedness_param_msg = ':param groundedness: bool'
            self.stacking_note = ''

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

        self.ct = 0
        if self.proposal_config.deterministic:
            self.thetas = np.ones(1)
            self.theta_priors = np.ones(1)
            self.deltas = np.ones(1)
            self.delta_priors = np.ones(1)
        else:
            self.thetas = np.arange(0.5, 1.01, 0.01)
            self.theta_priors = np.exp(np.asarray([norm.logpdf(theta, self.config.agent.theta_mean, self.config.agent.theta_std) for theta in self.thetas]))
            self.deltas = np.arange(0.5, 1.01, 0.01)
            self.delta_priors = np.exp(np.asarray([norm.logpdf(delta, self.config.agent.delta_mean, self.config.agent.delta_std) for delta in self.deltas]))
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

            log.info([f'Cost at iteration {len(c)}', self.llm.get_info(), self.llm_exp.get_info(), self.poor_llm.get_info(), len(self.cache_h_program), self.ct])

            c = c.append_and_retnew(query, verdict)

        res = None
        prob_res = None
        if test_set is not None:
            res, prob_res, support, phat_h_given_c = asyncio.run(self.a_eval_test_set(c, test_set))
        
        log.info(['Final cost', self.llm.get_info(), self.llm_exp.get_info(), self.poor_llm.get_info(), len(self.cache_h_program), self.ct])

        return res, prob_res, support, phat_h_given_c
        

    async def a_score_h(self, h): # score by number of words 
        if h in self.cache_prior_h:
            return self.cache_prior_h[h]
        
        # outputs = await self.poor_llm.aprompt([prior_prompt.format(h=h)], temperature=0)
        # res = len(outputs[0].split(','))
        # self.cache_prior_h[h] = 1 / res
        # self.cache_prior_h[h] = 1 / (len(h.split(' ')) ** 2)
        self.cache_prior_h[h] = 1 / len(h.split(' '))
        return self.cache_prior_h[h]

    def eval_y_given_xprog(self, y, x, prog, thetas, deltas):
        by = (y == 'yes')
        if prog is None:
            return np.full(len(thetas) * len(deltas), 0.50)
        try:
            exec(prog)
            res = locals()[f'rule'](x)
            thetas = np.asarray(thetas)
            if not res:
                if by == res:
                    return np.repeat(deltas, len(thetas))
                else:
                    return np.repeat(1. - deltas, len(thetas))
                # return int(by == res)
            else:
                # return 0.5
                if by:
                    return np.tile(thetas, len(deltas))
                else:
                    return np.tile(1. - thetas, len(deltas))
        except: 
            return np.full(len(thetas) * len(deltas), 0.50)

    async def a_h2prog(self, h):
        if h in self.cache_h_program:
            return self.cache_h_program[h]
        if h == "I don't know":
            self.cache_h_program[h] = None
        else:
            outputs = await self.poor_llm.aprompt([rule_translation_prompt.format(h=h, groundedness_param_msg=self.groundedness_param_msg)], temperature=0, seed=0)
            self.ct += 1
            self.cache_h_program[h] = outputs[0]
        return self.cache_h_program[h]
    
    def objective(self, theta, prog, c):
        theta = theta[0]
        args_list = [(y, x, prog, [theta], [0.9]) for x, y in zip(c.structures, c.labels)]
        y_given_xh_res = [self.eval_y_given_xprog(*args)[0] for args in args_list]
        if np.prod(y_given_xh_res) > 0:
            log_prior = norm.logpdf(theta, 0.6, 0.1)
            return -(log_prior + np.sum(np.log(y_given_xh_res)))
        else:
            return 30
    
    def a_find_best_theta(self, prog, c):
        bounds = Bounds(lb = 0.5, 
            ub = 1.0)

        options = {
            'maxfun': 50
        }

        res = minimize(self.objective, 
            0.7, 
            args=(prog, c),
            method='L-BFGS-B', 
            bounds=bounds,
            options=options)
        return res.x[0]
    
    
    async def a_score_joints(self, hs, c, noprior=False):
        unique_hs = list(set(hs))
        outputs = await asyncio.gather(*[self.a_h2prog(h) for h in unique_hs])
        h2prog = {h: prog for h, prog in zip(unique_hs, outputs)}

        res = []

        for h in hs:
            args_list = [(y, x, h2prog[h], self.thetas, self.deltas) for x, y in zip(c.structures, c.labels)]
            y_given_xh_res = np.asarray([self.eval_y_given_xprog(*args) for args in args_list])
            sm = np.sum(np.prod(y_given_xh_res, axis=0) * self.all_priors)
            if noprior:
                res.append(sm)
            else:
                prior = await self.a_score_h(h)
                res.append(prior * sm)
        return res
    
    async def a_dist_h_given_c_with_support(self, c, support):
        joints = await self.a_score_joints(support, c)
        if np.sum(joints) == 0: # This is possible if support is totally inconsistent with cs
            return np.zeros_like(joints)
        return np.asarray(joints) / np.sum(joints)
    
    async def a_sample_proposal_q(self, c):
        if c.to_text() in self.cache_c:
            return self.cache_c[c.to_text()][:self.config.agent.num_hs]

        if self.proposal_config.name.startswith('is'): 
            res = await self.a_importance_sampling(c)
        elif self.proposal_config.name == 'particle_filter': # TODO: Be super careful with how many times this is called
            res = await self.a_particle_filter(c)
        else:
            raise NotImplementedError
        self.cache_c[c.to_text()] = res
        return self.cache_c[c.to_text()][:self.config.agent.num_hs]

    async def a_importance_sampling(self, c):
        hs_dict = {}
        # Reuse from previous turn ---
        if not self.first_propose:
            previous_c = ZendoGame(c.structures[:-1], c.labels[:-1])
            potential_hs = self.cache_c[previous_c.to_text()]
            probs = await self.a_score_joints(potential_hs, c) 
            probs = np.asarray(probs, dtype=np.float64)
            for h, p in zip(potential_hs, probs):
                if p > 0:
                    hs_dict[h] = p
            log.info(f'Still valid hypotheses from previous round:\n{list(hs_dict)}')
        # ----------------------------

        self.first_propose = False

        # Thoughts: Could speed this up for IS but will still be tricky for IS-Refine which is more sequential
        ct = 0
        while (ct < self.proposal_config.num_max_calls_per_it) and len(hs_dict) < 30:
            ct += 1

            internal_ct = 0
            while internal_ct == 0 or (internal_ct < 20 and sub_c.to_text() in self.cache_sub_c):
                x1, y1, x2, y2 = c.samp_two()
                if x2 is None:
                    sub_c = ZendoGame([x1], [y1])
                else:
                    sub_c = ZendoGame([x1, x2], [y1, y2])
                internal_ct += 1

            potential_hs = await self.a_sample_potential_hs(sub_c)

            # log.info(f'Potential hs: {potential_hs}')
            
            probs = await self.a_score_joints(potential_hs, c)
            probs = np.asarray(probs, dtype=np.float64)

            # For refinement
            if self.proposal_config.name == 'is_refine':
                refine_ct = 0
                # Thoughts: can only easily decide when to refine when things are deterministic. Less clear when to refine when things are not deterministic.
                while np.sum(probs) == 0 and refine_ct < self.proposal_config.num_refine_it and (ct < self.proposal_config.num_max_calls_per_it):
                    potential_hs = await self.a_refine_hs(potential_hs, c)

                    # log.info(f'REFINED {refine_ct} Potential hs: {potential_hs}')
            
                    probs = await self.a_score_joints(potential_hs, c)
                    probs = np.asarray(probs, dtype=np.float64)
                    refine_ct += 1
                    ct += 1

            for h, p in zip(potential_hs, probs):
                if p > 0:
                    hs_dict[h] = p
            log.info(f'Iteration {ct}: got {len(hs_dict)} hs')
            # log.info(f'Valid hypotheses:\n{list(hs_dict)}')

        res = np.random.permutation(list(hs_dict))
        log.info(f'Current working rules\n{res}')
        # log.info(f'PROBS {hs_dict}')

        if len(res) == 0:
            log.info(f'NO WORKING RULES -- reusing bad hypotheses from last iteration')
            res = self.cache_c[previous_c.to_text()]

        return res
    
    def deduplicate_particles(self, particles):
        return [p.split('##')[-1] for p in particles]

    def duplicate_particles(self, particles, num, single=False):
        if single: 
            x = particles
            return[f'{id}##' + x for id in range(num)]
        return sum([[f'{id}##' + x for id in range(num)] for x in particles], [])
    
    async def a_particle_filter(self, c):
        if len(c) != self.pf_checkpoint + 1:
            raise Exception('Should not be here')
        else:
            self.pf_checkpoint += 1

        old_particles, old_particle_weights = self.particles, self.particle_weights

        if self.first_propose:
            good_x, _ = c.samp_good_bad()
            sub_c = good_x

            potential_hs = await self.a_sample_potential_hs(sub_c)
            
            probs = await self.a_score_joints(potential_hs, c)
            probs = np.asarray(probs, dtype=np.float64)
            
            self.particles = np.asarray(potential_hs, dtype='object')
            self.particle_weights = np.asarray(probs)

            self.particle_weights = self.particle_weights / np.sum(self.particle_weights)

        else:
            evolve_indices = []
            to_evolve = {}

            # ------------- Rejuvenate -----------------
            log.info('Rejuvenation...')

            # Choose what to rejuvenate
            tplusone_likelihoods = await self.a_score_joints(self.particles, c, noprior=True)
            sorted_indices = np.argsort(tplusone_likelihoods)
            for idx in sorted_indices:
                if self.proposal_config.num_max_calls_per_it == len(to_evolve) and self.particles[idx] not in to_evolve:
                    continue
                if tplusone_likelihoods[idx] == 1: # In deterministic case, don't rejuvenate perfectly fine rules
                    continue
                evolve_indices.append(idx)
                to_evolve[self.particles[idx]] = True
            to_evolve = list(to_evolve.keys())

            # Get rejuvenation options
            evolve_options_list = await asyncio.gather(*[self.a_get_rejuvenation_options(h, 
                                                                                         c.structures[-1], 
                                                                                         c.labels[-1], 
                                                                                         c) for h in to_evolve])
            evolve_options_dict = dict(zip(to_evolve, evolve_options_list))

            # Rejuvenate
            new_particles = []
            probs = await self.a_score_joints(to_evolve, c)
            for (k, (options, option_scores)), p in zip(evolve_options_dict.items(), probs):
                options, option_scores = np.asarray(options), np.asarray(option_scores)
                options, option_scores = options[option_scores > p], option_scores[option_scores > p]

                # make sure to include itself
                # options = np.append(options, k)
                # option_scores = np.append(option_scores, p)

                options_prob = np.asarray(option_scores) / sum(option_scores)
                
                # Hand engineered conditions
                if len(options_prob) > self.config.agent.n_neighbors:
                    indices = systematic_resample(options_prob, self.config.agent.n_neighbors) # TODO: can use down sampling without replacement instead
                    new_particles.append(options[indices])
                    log.info(f"Rejuvenate '{k}' to '{options[indices]}'")
                elif len(options_prob) > 0:
                    new_particles.append(options)
                    log.info(f"Rejuvenate '{k}' to '{options}'")
                # if len(options_prob) > 0:
                #     new_particles.append(options)
                #     log.info(f"Rejuvenate '{k}' to '{options}'")
                else: 
                    log.info(f"Rejuvenate '{k}' to 'NOTHING'")
                new_particles.append([k])
            
            # Upsample the rest
            if len(to_evolve) == 0:
                new_particles = [self.particles]
            else:
                # avg_upsample = len(new_particles) // len(to_evolve) # make sure this is never going to be 0
                avg_upsample = 1
                new_particles = np.unique(np.concatenate(new_particles))
                new_particles = [self.duplicate_particles(new_particles, 1)]
                for p in self.particles:
                    if p not in to_evolve:
                        new_particles = new_particles + [self.duplicate_particles(p, avg_upsample, single=True)]
            
            self.particles = np.unique(np.concatenate(new_particles))
            self.particle_weights = np.ones(len(self.particles)) / len(self.particles)
            # ------------- End rejuvenate -----------------
            
            log.info('Reweighting...')
            new_joints = await self.a_score_joints(self.deduplicate_particles(self.particles), c)

            for idx, (h, new_joint) in enumerate(zip(self.particles, new_joints)):
                self.particle_weights[idx] *= new_joint

            if np.sum(self.particle_weights) == 0:
                self.particle_weights[:] = 1
            self.particle_weights = self.particle_weights / np.sum(self.particle_weights)
        
        self.first_propose = False

        log.info('Resampling...')
        resampled_indices = systematic_resample(self.particle_weights, self.proposal_config.num_particles)

        # log.info('Before (down) resampling')
        # for idx, (h, p) in enumerate(zip(self.particles, self.particle_weights)):
        #     log.info(f'{idx} {h} {p}')
        # log.info(resampled_indices)

        self.particles = self.particles[resampled_indices]
        self.particle_weights = np.ones(len(self.particles)) / len(self.particles)
        self.particles = self.deduplicate_particles(self.particles)

        # Return unique, working particles
        hs_dict = {}
        for h, p in zip(self.particles, self.particle_weights):
            if p > 0:
                hs_dict[h] = p

        # Check
        old_hs_dict = {}
        for h in old_particles:
            old_hs_dict[h] = True
        
        log.info('Changes to particles:')
        for h in old_hs_dict:
            if h not in hs_dict:
                log.info(f'DELETED {h}')
        for h in hs_dict:
            if h not in old_hs_dict:
                log.info(f'ADDED {h}')

        res = np.random.permutation(list(hs_dict))
        log.info(f'Current working rules\n{pprint.pformat(list(res))}')
        return res
    
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
                                                                     num=self.proposal_config.num_hypotheses_per_call,
                                                                     att_par=self.att_par)], temperature=0, seed=self.config.seed)
        rules = sum([parse_listed_output(output) for output in outputs], [])
        rules = np.asarray([rule.split('->')[-1].strip(" '\n") for rule in rules])
        probs = await self.a_score_joints(rules, c)
        probs = np.asarray(probs, dtype=np.float64)
        return rules, probs

    async def a_count_score_hs(self, hs, c):
        unique_hs = list(set(hs))
        outputs = await asyncio.gather(*[self.a_h2prog(h) for h in unique_hs])
        h2prog = {h: prog for h, prog in zip(unique_hs, outputs)}

        all_res = []

        for h in hs:
            args_list = [(y, x, h2prog[h], [1], [1]) for x, y in zip(c.structures, c.labels)]
            y_given_xh_res = [self.eval_y_given_xprog(*args)[0] for args in args_list]
            all_res.append(y_given_xh_res)

        return all_res

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

        feedback = feedback_generator(fb_structures, fb_labels)

        outputs = await self.llm.aprompt([refine_prompt.format(h=best_h, feedback=feedback, num=self.proposal_config.num_hypotheses_per_call * 3,
                                                               att_par=self.att_par)], temperature=0, seed=self.config.seed)
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
            res = [ZendoStructure(random_block_rng=np.random.default_rng(np.random.randint(500)), stacking_flag=self.stacking_flag) for _ in range(self.config.agent.num_xs)]
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
                                                                    example_block=self.zendo_config.example_block,
                                                                    stacking_note=self.stacking_note)], temperature=0, seed=self.config.seed)
                                                            

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
                                                                          example_block=self.zendo_config.example_block,
                                                                          stacking_note=self.stacking_note)], temperature=0, seed=self.config.seed)

        cur_output = outputs[0]
        x = cur_output.split('Structure 1:', 1)[1].split('End of structure 1', 1)[0] + '\n'

        self.cache_hs[list_to_str(hs)] = x
        return self.cache_hs[list_to_str(hs)]
    
    async def a_get_kl_divergence(self, c, x, y):
        support = await self.a_sample_proposal_q(c)
        phat_h_given_cxy = await self.a_dist_h_given_c_with_support(c.append_and_retnew(x, y), support)
        phat_h_given_c = await self.a_dist_h_given_c_with_support(c, support)

        # If everything is equal to zero given the support, that means things are very different
        if np.all(phat_h_given_cxy == 0):
            return 1000000

        kl_divergence = 0
        for i in range(len(phat_h_given_cxy)):
            if phat_h_given_cxy[i] != 0: # Important
                kl_divergence += phat_h_given_cxy[i] * (np.log(phat_h_given_cxy[i]) - np.log(phat_h_given_c[i]))
        return kl_divergence
    
    async def a_dist_y_given_cx(self, c, x):
        if c.to_text() + x.to_text() in self.cache_cx:
            return self.cache_cx[c.to_text() + x.to_text()]
        dist_y_given_cx = []
        hs = await self.a_sample_proposal_q(c)
        # hs = np.concatenate((hs, ["I don't know"]))
        for y in ['yes', 'no']:
            joints = await self.a_score_joints(hs, c.append_and_retnew(x, y))
            dist_y_given_cx.append(np.sum(joints))
        if np.sum(dist_y_given_cx) == 0: # happens when no hypotheses is consistent with everything 
            dist_y_given_cx = np.ones_like(dist_y_given_cx)
        dist_y_given_cx = np.asarray(dist_y_given_cx) / np.sum(dist_y_given_cx)
        self.cache_cx[c.to_text() + x.to_text()] = dist_y_given_cx
        return self.cache_cx[c.to_text() + x.to_text()]
    
    async def a_get_expected_kl_divergence(self, c, x):
        dist_y_given_cx = await self.a_dist_y_given_cx(c, x)

        if np.all(dist_y_given_cx == 0):
            return 0

        kl_divergences = await asyncio.gather(self.a_get_kl_divergence(c, x, 'yes'), self.a_get_kl_divergence(c, x, 'no'))
        
        if np.max(kl_divergences) == 1000000:
            return 1e-6
        
        return np.sum(dist_y_given_cx * np.asarray(kl_divergences))
    
    async def a_get_relevant_xs(self, c):
        hs = await self.a_sample_proposal_q(c)
        log.info(f'Coming up with xs...')
        res = await self.a_sample_proposal_r_handler(hs)
        filtered_res = [x for x in res if len(x) > 0]
        return filtered_res
    
    async def a_get_query_x(self, c):
        xs = await self.a_get_relevant_xs(c)
        
        # log.info(f'xs below')
        # for idx, x in enumerate(xs):
        #     log.info(f'{idx + 1}. \n{x.to_text()}')

        expected_kl_divergences = await asyncio.gather(*[self.a_get_expected_kl_divergence(c, x) for x in xs])
        # expected_kl_divergences = [await self.a_get_expected_kl_divergence(c, x) for x in xs] # TODO: Only doing this to slow things down
        # log.info(f'Expected kl divergence {expected_kl_divergences}')
        best_idx = np.argmax(expected_kl_divergences)

        log.info(f'Most discriminating structure: {best_idx+1}\n {xs[best_idx].to_text()}')
        log.info(f'Best expected kl divergence {expected_kl_divergences[best_idx]}')
        # # -------
        # for idx, x in enumerate(xs):
        #     log.info(f'{idx + 1}. \n{x.to_text()}')
        #     support = await self.a_sample_proposal_q(c)
        #     phat_h_given_c = await self.a_dist_h_given_c_with_support(c, support)
        #     phat_h_given_cxyes = await self.a_dist_h_given_c_with_support(c.append_and_retnew(x, 'yes'), support)
        #     phat_h_given_cxno = await self.a_dist_h_given_c_with_support(c.append_and_retnew(x, 'no'), support)
        #     dist_y_given_cx = await self.a_dist_y_given_cx(c, x)
        #     log.info(f'support {support}')
        #     log.info(f'phat_h_given_c {phat_h_given_c}')
        #     log.info(f'phat_h_given_cxyes {phat_h_given_cxyes}')
        #     log.info(f'phat_h_given_cxno {phat_h_given_cxno}')
        #     log.info(f'dist_y_given_cx {dist_y_given_cx}')
        # # -------
        if expected_kl_divergences[best_idx] == 0:
            log.info('NOTE Expected kl divergence is 0, trying random queries now')
            best_idx = np.random.choice(len(xs))
        #     hs = await self.a_sample_proposal_q(c)
        #     return hs[0], True
        return xs[best_idx], False

    async def a_eval_test_set(self, c, test_set, get_posterior=False):
        support = await self.a_sample_proposal_q(c)
        predicted_y_dists = await asyncio.gather(*[self.a_dist_y_given_cx(c, x) for x in test_set])
        log.info(f'predicted y dist {predicted_y_dists}')
        phat_h_given_c = await self.a_dist_h_given_c_with_support(c, support)

        # histogram extra
        histograms = np.zeros(9)
        for h, phat in zip(support, phat_h_given_c):
            score = 0
            for i, x in enumerate(test_set):
                y_dist = np.zeros(2)
                for idx, y in enumerate(['yes', 'no']):
                    joints = await self.a_score_joints([h], c.append_and_retnew(x, y))
                    y_dist[idx] = joints[0]
                y_dist = y_dist / np.sum(y_dist)
                if i < 4 and y_dist[0] > y_dist[1]:
                    score += 1
                elif i >= 4 and y_dist[0] <= y_dist[1]:
                    score += 1
            histograms[score] += phat
        log.info(f'histograms: {list(histograms)}')


        return [bool(y_dist[0] > y_dist[1]) for y_dist in predicted_y_dists], [y_dist[0] for y_dist in predicted_y_dists], support, phat_h_given_c