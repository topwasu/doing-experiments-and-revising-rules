import logging

from prompts.zendo import *
from toptoolkit.llm import create_llm
from data.zendo import ZendoGame, ZendoBlock, ZendoStructure


log = logging.getLogger(__name__)


class LLMNaiveZendo:
    def __init__(self, config, zendo_config, stacking_flag=False):
        self.config = config
        self.zendo_config = zendo_config
        self.llm = create_llm('gpt-4-1106-preview')
        # self.llm.setup_cache('disk')
        if stacking_flag:
            self.att_par = """color (blue/red/green) 
size (small/medium/large)
orientation (upright/left/right/strange)
groundedness (grounded/ungrounded/stacking),
touching (which other blocks they do and do not touch)."""
        else:
            self.att_par = """color (blue/red/green) 
size (small/medium/large)
orientation (upright/left/right/strange)
groundedness (grounded/ungrounded),
touching (which other blocks they do and do not touch)."""

    def play_zendo(self, moderator, game, test_set=None):
        conversation = [play_zendo_hard_prompt.format(text_c=game.to_text(), att_par=self.att_par)]
        ct = 1
        while ct < 8:
            summary = self.llm.prompt([conversation], temperature=0, seed=self.config.seed)[0]
            conversation.append(summary)
            conversation.append('Give one structure you want to test whether it follows the secret rule or not. Do not include anything other than the structure.')
            query = self.llm.prompt([conversation], temperature=0, seed=self.config.seed)[0]

            log.info(summary)
            log.info(query)
            log.info(f'Parsed query {ZendoStructure(txt=query).to_text()}')
            verdict = moderator.query(ZendoStructure(txt=query))
            log.info(f'LLM moderator verdict: {verdict}')

            conversation.append(query)
            conversation.append(f'The verdict on whether the queried structure follows the rule is {verdict}. Give a very short summary on what you currently think the secret rule is.')
            ct += 1
        
        conversation = conversation[:-1]
        res = None
        prob_res = None
        if test_set is not None:
            guesses = self.llm.prompt([conversation + [f'Now, do you think this structure follow the rule?\n: {x.to_text()}\nAnswer only yes or no. Give your best guess even if you are uncertain. Do not explain. Just say yes or no'] for x in test_set], temperature=0)
            res = [True if guess.lower().strip(' .\n') == 'yes' else False for guess in guesses]
            prob_res = [1. if guess.lower().strip(' .\n') == 'yes' else 0. for guess in guesses]
        return res, prob_res