import asyncio
import logging
import numpy as np
import pprint
from abc import ABC, abstractmethod
from scipy.optimize import minimize, Bounds
from scipy.stats import norm

from data.acre import ACREGame, ACREGroup, ACREObject
from data.zendo import ZendoGame, ZendoStructure, ZendoBlock

from .utils import systematic_resample


log = logging.getLogger(__name__)
# log.level = logging.DEBUG


class LLMScientist(ABC):
    @abstractmethod
    def play(self, moderator, game, test_set):
        pass

    async def a_score_h(self, h):
        if h in self.cache_prior_h:
            return self.cache_prior_h[h]
        
        self.cache_prior_h[h] = 1 / len(h.split(' '))
        return self.cache_prior_h[h]

    def eval_y_given_xprog(self, y, x, prog, thetas, deltas):
        by = (y == self.possible_ys[0])
        if prog is None:
            return np.full(len(thetas) * len(deltas), 0.50)
        try:
            exec(prog) # Need to make sure we import the class def we need here
            res = locals()[f'rule'](x)
            thetas = np.asarray(thetas)
            if not res:
                if by == res:
                    return np.repeat(deltas, len(thetas))
                else:
                    return np.repeat(1. - deltas, len(thetas))
            else:
                if by:
                    return np.tile(thetas, len(deltas))
                else:
                    return np.tile(1. - thetas, len(deltas))
        except Exception as e:
            # log.debug(f"Exception while trying to evaluate {prog}:\n{e}")
            return np.full(len(thetas) * len(deltas), 0.50)

    async def a_h2prog(self, h):
        if h in self.cache_h_program:
            return self.cache_h_program[h]
        if h == "I don't know":
            self.cache_h_program[h] = None
        else:
            outputs = await self.poor_llm.aprompt([self.rule_translation_prompt.format(h=h)], temperature=0, seed=0)
            self.cache_h_program[h] = outputs[0]
        return self.cache_h_program[h]
    
    def objective(self, theta, prog, c):
        theta = theta[0]
        args_list = [(y, x, prog, [theta], [0.9]) for x, y in zip(c.xs, c.ys)]
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
            args_list = [(y, x, h2prog[h], self.thetas, self.deltas) for x, y in zip(c.xs, c.ys)]
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
    
    @abstractmethod
    def get_previous_c(self, c):
        pass
    
    @abstractmethod
    def get_sub_c(self, c):
        pass

    async def a_importance_sampling(self, c):
        hs_dict = {}
        # Reuse from previous turn ---
        if not self.first_propose:
            previous_c = self.get_previous_c(c)
            potential_hs = self.cache_c[previous_c.to_text()]
            probs = await self.a_score_joints(potential_hs, c) 
            probs = np.asarray(probs, dtype=np.float64)
            for h, p in zip(potential_hs, probs):
                if p > 0:
                    hs_dict[h] = p
            log.info(f'Still valid hypotheses from previous round:\n{list(hs_dict)}')
        # ----------------------------

        self.first_propose = False

        if self.proposal_config.name == 'is':
            potential_hs_lst = await asyncio.gather(*[self.a_sample_potential_hs(self.get_sub_c(c)) for _ in range(self.proposal_config.num_max_calls_per_it)])
                
            for ct, potential_hs in enumerate(potential_hs_lst):
                probs = await self.a_score_joints(potential_hs, c)
                probs = np.asarray(probs, dtype=np.float64)

                for h, p in zip(potential_hs, probs):
                    if p > 0:
                        hs_dict[h] = p
                log.info(f'Iteration {ct}: got {len(hs_dict)} hs')
                log.info(f'Valid hypotheses:\n{list(hs_dict)}')
            
        # Thoughts: Could speed this up for IS but will still be tricky for IS-Refine which is more sequential
        elif self.proposal_config.name == 'is_refine':
            ct = 0
            while (ct < self.proposal_config.num_max_calls_per_it) and len(hs_dict) < 30:
                ct += 1

                sub_c = self.get_sub_c(c)

                potential_hs = await self.a_sample_potential_hs(sub_c)

                log.info(f'Potential hs: {potential_hs}')
                
                probs = await self.a_score_joints(potential_hs, c)
                probs = np.asarray(probs, dtype=np.float64)

                # For refinement
                refine_ct = 0
                # Thoughts: can only easily decide when to refine when things are deterministic. Less clear when to refine when things are not deterministic.
                while np.sum(probs) == 0 and refine_ct < self.proposal_config.num_refine_it and (ct < self.proposal_config.num_max_calls_per_it):
                    potential_hs = await self.a_refine_hs(potential_hs, c)

                    log.info(f'REFINED {refine_ct} Potential hs: {potential_hs}')
            
                    probs = await self.a_score_joints(potential_hs, c)
                    probs = np.asarray(probs, dtype=np.float64)
                    refine_ct += 1
                    ct += 1

                for h, p in zip(potential_hs, probs):
                    if p > 0:
                        hs_dict[h] = p
                log.info(f'Iteration {ct}: got {len(hs_dict)} hs')
                log.info(f'Valid hypotheses:\n{list(hs_dict)}')

        res = np.random.permutation(list(hs_dict))
        log.info(f'Current working rules\n{res}')
        log.info(f'PROBS {hs_dict}')

        if len(res) == 0:
            log.info(f'NO WORKING RULES -- reusing bad hypotheses from last iteration')
            res = self.cache_c[previous_c.to_text()]

        return res
    
    @abstractmethod
    async def first_propose_hs(self, c):
        pass
    
    async def a_particle_filter(self, c):
        if len(c) != self.pf_checkpoint + 1:
            raise Exception('Should not be here')
        else:
            self.pf_checkpoint += 1

        old_particles, old_particle_weights = self.particles, self.particle_weights

        if self.first_propose:
            potential_hs = await self.first_propose_hs(c)
            
            probs = await self.a_score_joints(potential_hs, c)
            probs = np.asarray(probs, dtype=np.float64)
            
            self.particles = np.asarray(potential_hs, dtype='object')
            self.particle_weights = np.asarray(probs)

            log.debug(self.particles)

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
                                                                                         c.xs[-1], 
                                                                                         c.ys[-1], 
                                                                                         c) for h in to_evolve])
            evolve_options_dict = dict(zip(to_evolve, evolve_options_list))

            # Rejuvenate
            new_particles = []
            probs = await self.a_score_joints(to_evolve, c)
            for (k, (options, option_scores)), p in zip(evolve_options_dict.items(), probs):
                options, option_scores = np.asarray(options), np.asarray(option_scores)
                options, option_scores = options[option_scores > p], option_scores[option_scores > p] # Filter bad moves
                options_prob = np.asarray(option_scores) / sum(option_scores)
                
                # Hand engineered conditions
                if len(options_prob) > self.config.agent.n_neighbors:
                    indices = systematic_resample(options_prob, self.config.agent.n_neighbors)
                    new_particles.append(options[indices])
                    log.info(f"Rejuvenate '{k}' to '{options[indices]}'")
                elif len(options_prob) > 0:
                    new_particles.append(options)
                    log.info(f"Rejuvenate '{k}' to '{options}'")
                else: 
                    log.info(f"Rejuvenate '{k}' to 'NOTHING'")
            
            # Upsample the rest
            new_particles = new_particles + [self.particles]
            
            self.particles = np.unique(np.concatenate(new_particles))
            self.particle_weights = np.ones(len(self.particles)) / len(self.particles)
            # ------------- End rejuvenate -----------------
            
            log.info('Reweighting...')
            new_joints = await self.a_score_joints(self.particles, c)

            for idx, (h, new_joint) in enumerate(zip(self.particles, new_joints)):
                self.particle_weights[idx] *= new_joint

            if np.sum(self.particle_weights) == 0:
                self.particle_weights[:] = 1
            self.particle_weights = self.particle_weights / np.sum(self.particle_weights)
        
        self.first_propose = False

        log.info('Resampling...')
        resampled_indices = systematic_resample(self.particle_weights, self.proposal_config.num_particles)

        log.debug(f'Weights {self.particle_weights} {len(self.particle_weights)} Num_particles {self.proposal_config.num_particles}')

        # log.info('Before (down) resampling')
        # for idx, (h, p) in enumerate(zip(self.particles, self.particle_weights)):
        #     log.info(f'{idx} {h} {p}')
        # log.info(resampled_indices)

        self.particles = self.particles[resampled_indices]
        self.particle_weights = np.ones(len(self.particles)) / len(self.particles)

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
    
    @abstractmethod
    async def a_sample_potential_hs(self, sub_c):
        pass

    @abstractmethod
    async def a_get_rejuvenation_options(self, h, x, y, c):
        pass

    async def a_count_score_hs(self, hs, c):
        unique_hs = list(set(hs))
        outputs = await asyncio.gather(*[self.a_h2prog(h) for h in unique_hs])
        h2prog = {h: prog for h, prog in zip(unique_hs, outputs)}

        all_res = []

        for h in hs:
            args_list = [(y, x, h2prog[h], [1], [1]) for x, y in zip(c.xs, c.ys)]
            y_given_xh_res = [self.eval_y_given_xprog(*args)[0] for args in args_list]
            all_res.append(y_given_xh_res)

        return all_res

    @abstractmethod
    async def a_refine_hs(self, hs, c):
        pass
    
    @abstractmethod
    async def a_sample_proposal_r_handler(self, hs):
        pass
    
    async def a_get_kl_divergence(self, c, x, y):
        support = await self.a_sample_proposal_q(c)
        phat_h_given_cxy = await self.a_dist_h_given_c_with_support(c.append_and_retnew(x, y), support)
        phat_h_given_c = await self.a_dist_h_given_c_with_support(c, support)

        # If everything is equal to zero given the support, that means things are very different
        if np.all(phat_h_given_cxy == 0):
            return 1000000
        
        log.debug('YO')
        log.debug(phat_h_given_cxy)
        log.debug(phat_h_given_c)

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
        for y in self.possible_ys:
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

        kl_divergences = await asyncio.gather(self.a_get_kl_divergence(c, x, self.possible_ys[0]), self.a_get_kl_divergence(c, x, self.possible_ys[1]))
        
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
        if expected_kl_divergences[best_idx] == 0:
            log.info('NOTE Expected kl divergence is 0, trying random queries now')
            best_idx = np.random.choice(len(xs))
        #     hs = await self.a_sample_proposal_q(c)
        #     return hs[0], True
        return xs[best_idx], False

    @abstractmethod
    async def a_eval_test_set(self, c, test_set):
        pass