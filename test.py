import asyncio
import logging
import numpy as np

from agents.zendo import LLMScientistZendo
from data.zendo import ZendoBlock, ZendoStructure, ZendoGame, AdvZendoConfig, AdvZendoConfigStacking
from toptoolkit.llm import create_llm
from toptoolkit.logging.logging import init_logger

import hydra
from omegaconf import OmegaConf
from pathlib import Path


init_logger('info')
log = logging.getLogger(__name__)


def test_rep(config):
    xs = [
        # ZendoStructure([ZendoBlock('blue', 'small', 'upright'), ZendoBlock('blue', 'medium', 'upright')]),
        ZendoStructure([ZendoBlock('blue', 'small', 'upright'), 
                        ZendoBlock('red', 'medium', 'upright'),
                        ZendoBlock('green', 'large', 'flat'),
                        ZendoBlock('yellow', 'small', 'flat')]),
        ZendoStructure([ZendoBlock('blue', 'large', 'upright'), 
                        ZendoBlock('red', 'medium', 'flat'),
                        ZendoBlock('green', 'small', 'upright')])
    ]
    ys = ['yes', 'no']
    log.info(xs[0].to_text())
    log.info(xs[0].to_text('color'))
    c = ZendoGame(xs, ys)
    log.info(c.to_text())
    log.info(c.to_text('color'))


def test_proposal_q(config):
    xs = [
        # ZendoStructure([ZendoBlock('blue', 'small', 'upright'), ZendoBlock('blue', 'medium', 'upright')]),
        ZendoStructure([ZendoBlock('blue', 'small', 'upright'), 
                        ZendoBlock('red', 'medium', 'upright'),
                        ZendoBlock('green', 'large', 'flat'),
                        ZendoBlock('yellow', 'small', 'flat')]),
        ZendoStructure([ZendoBlock('blue', 'large', 'upright'), 
                        ZendoBlock('red', 'medium', 'flat'),
                        ZendoBlock('green', 'small', 'upright')])
    ]
    ys = ['yes', 'no']
    c = ZendoGame(xs, ys)
    

    agent = LLMScientistZendo(config, AdvZendoConfig)
    log.info(c.to_text('color'))
    print(asyncio.run(agent.a_sample_proposal_q(c)))


def test_eval_y_given_xh(config):
    x = ZendoStructure([ZendoBlock('red', 'large', 'right', True, []), 
                        ZendoBlock('blue', 'medium', 'left', False, [])])
    agent = LLMScientistZendo(config, AdvZendoConfig)
    hs = [
        # 'There is at least one red block.'
        'There are no upright-oriented blocks.',
        'There is no upright-oriented block',
        'The number of upright blocks is zero'
        # 'Color: At least one block must be a color other than blue.',
        # 'Color: At least one block must be blue.',
        # 'Touching: Not all blocks are touching each other.',
        # 'Groundedness: There cannot be more than one ungrounded block.'
    ]
    for h in hs:
        print(asyncio.run(agent.a_eval_y_given_xh('yes', x, h)))
        print(asyncio.run(agent.a_eval_y_given_xh('no', x, h)))
        # Should get 0, 1, 0, 1

def test_proposal_r(config):
    agent = LLMScientistZendo(config, AdvZendoConfigStacking, stacking_flag=True)
    hs = [
        # 'Color: At least one block must be a color other than blue.',
        # 'Color: At least one block must be blue.',
        'There is a stacking block',
        # 'Groundedness: There cannot be more than one ungrounded block.'
    ]
    xs = asyncio.run(agent.a_sample_proposal_r_handler(hs))
    print(hs)
    for x in xs:
        print(x.to_text())


def test_proposal_prompt(config):
#     text_c = """Good structure 1:
# - Block 1: Color - blue, Size - small
# - Block 2: Color - red, Size - medium
# - Block 3: Color - green, Size - large
# - Block 4: Color - green, Size - small

# Bad structure 1:
# - Block 1: Color - blue, Size - large
# - Block 2: Color - red, Size - medium
# - Block 3: Color - green, Size - small
# """
    text_c = """Good structure 1:
- Block 1: Groundedness - grounded, Color - blue, Size - small, Orientation - left, Touching - None
- Block 2: Groundedness - ungrounded, Color - red, Size - medium, Orientation - flat, Touching - None
- Block 3: Groundedness - ungrounded, Color - green, Size - small, Orientation - upright, Touching - None

Bad structure 1:
- Block 1: Groundedness - ungrounded, Color - blue, Size - large, Orientation - upright, Touching - None
- Block 2: Groundedness - ungrounded, Color - red, Size - medium, Orientation - flat, Touching - None
- Block 3: Groundedness - ungrounded, Color - green, Size - small, Orientation - upright, Touching - None"""

    text_c = """Bad structure 1:
- Block 1: Groundedness - ungrounded, Color - blue, Size - large, Orientation - upright, Touching - None
- Block 2: Groundedness - ungrounded, Color - red, Size - medium, Orientation - flat, Touching - None
- Block 3: Groundedness - ungrounded, Color - green, Size - small, Orientation - upright, Touching - None"""
    
    from prompts.zendo import propose_h_all_prompt
    llm = create_llm('gpt-4')
    llm.setup_cache('disk')
    for seed in range(1):
        res = llm.prompt([propose_h_all_prompt.format(text_c=text_c, att_summary='color (blue/red/green), size (small/medium/large), orientation (upright/left/right/strange), groundedness (grounded/ungrounded), touching (none/blocks it touch)', num=10)], 
                            temperature=0.7,
                            seed=seed)
        log.info(res)


def test_evolve(config): 
    # h = 'All blocks in a good structure must be touching each other.'
    # h = 'All blocks in a good structure must be touching each other and blocks of different colors (blue/red/green) should not be touching each other.'
    # h = 'There must be a green block'
    # h = 'All large blocks in a good structure must be blue.'
    # h = 'The good structure has at least one blue block'
    h = 'There is at least one block'
    # h = 'All blocks are blue'
    # x = ZendoStructure([ZendoBlock('green', 'large', 'upright', False, [2]), 
    #                     ZendoBlock('green', 'small', 'flat', False, [1])])
    # y = 'no'
    # x = ZendoStructure([ZendoBlock('blue', 'large', 'flat', False, [2]), 
    #                     ZendoBlock('green', 'medium', 'flat', True, [1])])
    x = ZendoStructure([ZendoBlock('green', 'small', 'upright', True, [])])
    # x = ZendoStructure([ZendoBlock('blue', 'large', 'flat', True, [1])])
    y = 'no'
    agent = LLMScientistZendo(config, AdvZendoConfig)
    modified_h = asyncio.run(agent.a_get_rejuvenation_options(h, x, y, c=ZendoGame([x], [y])))
    log.info(f'Modified hypothesis {modified_h}')


def test_basic_proposal(config):
    config.agent.proposal = 'particle_filter'
    x = ZendoStructure([ZendoBlock('blue', 'small', 'upright', True, []), 
                        ZendoBlock('blue', 'medium', 'upright', True, [3]),
                        ZendoBlock('green', 'small', 'upright', True, []),
                        ZendoBlock('green', 'medium', 'upright', True, [1]),
                        ZendoBlock('green', 'large', 'upright', True, [])])
    agent = LLMScientistZendo(config, AdvZendoConfig)
    hs = asyncio.run(agent.a_sample_potential_hs(x, 5))
    log.info(hs)

def test_synthesis(config):
    from prompts.zendo import rule_translation_prompt
    llm = create_llm('gpt-3.5-turbo')
    # llm.setup_cache('disk')
    # hs = ["There must be exactly three blocks touching the ground", "There is no block that is not in contact with the ground.", "Every block is the same size", "no block is upright", "there is a block touching another block", "There's a blue block touching a red block"]
    # hs = ["There is at least one medium block that is not green and is touching two blue blocks."]
    hs = ["The number of large blocks is greater than the number of blocks of any other size."]
    outputs = llm.prompt([rule_translation_prompt.format(h=h) for h in hs], temperature=0)
    log.info(outputs)
    log.info(['Cost', llm.get_info()])

def test_refine(config):
    xs = [
        # ZendoStructure([ZendoBlock('blue', 'small', 'upright'), ZendoBlock('blue', 'medium', 'upright')]),
        ZendoStructure([ZendoBlock('blue', 'small', 'upright', True, [2]), 
                        ZendoBlock('blue', 'medium', 'upright', True, [1]),
                        ZendoBlock('green', 'large', 'flat', False, [])]),
        ZendoStructure([ZendoBlock('blue', 'large', 'upright', False, [2]), 
                        ZendoBlock('red', 'medium', 'flat', False, [1]),
                        ZendoBlock('green', 'small', 'upright', False, [])])
    ]
    ys = ['no', 'yes']
    c = ZendoGame(xs, ys)

    hs = [
        'There is a medium block',
    ]

    # config.agent.proposal = 'is_refine'
    agent = LLMScientistZendo(config, AdvZendoConfig)

    hs = asyncio.run(agent.a_refine_hs(hs, c))
    log.info(hs)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    # test_rep(config)
    # test_proposal_q(config)
    # test_eval_y_given_xh(config)
    test_proposal_r(config)
    # test_proposal_prompt(config)
    # test_evolve(config)
    # test_basic_proposal(config)
    # test_synthesis(config)
    # test_refine(config)
#     x = ZendoStructure([ZendoBlock('blue', 'small', 'upright', True, [2]), 
#                         ZendoBlock('blue', 'medium', 'upright', True, [1, 3], 1),
#                         ZendoBlock('green', 'large', 'flat', False, [2])])
#     log.info(x.to_text())
#     x = ZendoStructure(txt="""Structure:
# - Block 1: Color - blue, Size - small, Orientation - upright, Groundedness - grounded, Touching - Block 2
# - Block 2: Color - blue, Size - medium, Orientation - upright, Groundedness - grounded, Touching - Block 1 (stacking), Block 3
# - Block 3: Color - green, Size - large, Orientation - flat, Groundedness - ungrounded, Touching - None""")
#     log.info(x.to_text())
    

if __name__ == '__main__':
    main()