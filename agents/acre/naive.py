import logging
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from prompts.acre import *
from openai_hf_interface import create_llm
from data.acre import ACREGroup, ACREGame


log = logging.getLogger(__name__)


class LLMNaiveACRE:
    def __init__(self, config, all_objects, llm=None):
        self.config = config
        self.all_objects = all_objects

        self.llm = llm
        cache_mode = 'disk_to_memory' if self.config.use_memory else 'disk'
        if self.llm is None: 
            self.llm = create_llm('gpt-4-1106-preview')
            self.llm.setup_cache(cache_mode, database_path=config.database_path)
            self.llm.set_default_kwargs({'timeout': 60, 'request_timeout': 60})

    def play(self, moderator, game, test_game=None):
        conversation = [play_prompt.format(text_c=game.to_text(), all_objects=', '.join(self.all_objects))]
        ct = 1
        while ct < 8:
            summary = self.llm.prompt([conversation], temperature=0, seed=self.config.seed)[0]
            conversation.append(summary)
            conversation.append('Give one group of objects you want to test whether the group follows the secret rule or not. Do not include anything other than the group of objects.')
            query = self.llm.prompt([conversation], temperature=0, seed=self.config.seed)[0]

            log.info(summary)
            log.info(query)
            log.info(f'Parsed query {ACREGroup(txt=query).to_text()}')
            good_output = True
            try:
                verdict = moderator.query(ACREGroup(txt=query))
            except:
                conversation.append(query)
                conversation.append(f"The group of objects '{query}' contains an object that is not available in the list of available objects {', '.join(self.all_objects)}. Please try again. Remember, do not include anything other than the group of objects (that are available).")
                query = self.llm.prompt([conversation], temperature=0, seed=self.config.seed)[0]
                try:
                    verdict = moderator.query(ACREGroup(txt=query))
                except:
                    good_output = False
            log.info(f'LLM moderator verdict: {verdict}')

            conversation.append(query)
            if good_output:
                conversation.append(f'The verdict on whether the queried group of objects follows the rule is {verdict}. Give a very short summary on what you currently think the secret rule is.')
            else:
                conversation.append(f'Whether the queried group of objects follows the rule cannot be determined since it contains some objects that are not available in our list of objects')
            ct += 1
        
        conversation = conversation[:-1]
        res = None
        prob_res = None
        if test_game is not None:
            guesses = self.llm.prompt([conversation + [f'Now, do you think this group of objects follow the rule?\n: {x.to_text()}\nAnswer only yes or no. Give your best guess even if you are uncertain. Do not explain. Just say yes or no'] for x in test_game.xs], temperature=0)
            res = [True if guess.lower().strip(' .\n') == 'yes' else False for guess in guesses]
            prob_res = [1. if guess.lower().strip(' .\n') == 'yes' else 0. for guess in guesses]

            correct_indices = [0 if y == 'on' else 1 for y in test_game.ys]
            roc_auc = roc_auc_score(correct_indices, 1 - np.asarray(prob_res))
            f1 = f1_score(correct_indices, 1 - np.asarray(prob_res), pos_label=0)
            task_solved = bool(f1 == 1.0)
            log.info(['Final cost', self.llm.get_info()])
        return res, prob_res, roc_auc, f1, task_solved