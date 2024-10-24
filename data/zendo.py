import json
import numpy as np
import random

from prompts.zendo import *
from openai_hf_interface import create_llm
from .zendo_rule_programs import *


class ZendoConfig:
    att = ['colors', 'sizes', 'orientations']
    att_choices = {
        'colors': 'blue/red/green/yellow',
        'sizes': 'small/medium/large',
        'orientations': 'upright/flat'
    }
    spec = "color (blue/red/green/yellow)\nsize (small/medium/large)\norientation (upright/flat)"
    example_block = "- Block 1: Color - color, Size - size, Orientation - orientation"


class AdvZendoConfig:
    att = ['colors', 'sizes', 'orientations', 'groundedness', 'touchings']
    att_choices = {
        'colors': 'blue/red/green',
        'sizes': 'small/medium/large',
        'orientations': 'upright/left/right/strange',
        'groundedness': 'grounded/ungrounded',
        'touchings': 'none/blocks it touch',
    }
    att_choices_list = {
        'colors': ['blue', 'red', 'green'],
        'sizes': ['small', 'medium', 'large'],
        'orientations': ['upright', 'left', 'right', 'strange'],
        'groundedness': [True, False],
    }
    spec = "color (blue/red/green)\nsize (small/medium/large)\norientation (upright/left/right/strange)\ngroundedness (grounded/ungrounded)\ntouching (none/blocks it touch)"
    example_block = "- Block 1: Color - color, Size - size, Orientation - orientation, Groundedness - groundedness, Touching - touching"

    def get_spec(rng=None): 
        txt = ""
        for k, v in AdvZendoConfig.att_choices.items():
            if rng is not None:
                v_list = v.split('/')
                v = '/'.join(rng.permutation(v_list))
            txt += f'{k} ({v})\n'
        return txt


def get_zendo_rules_and_examples():
    simple_rules = [
        # 'all its blocks are the same color.',
        # 'all its blocks are the same size.',
        # 'all its blocks are flat.',
        'it contains at least one red block.',
        'it contains at least one small block.',
        'it contains at least one block of each of the four colors.',
        'it contains no green blocks.',
        'it contains no large blocks.',
        # 'it contains at least one medium yellow block.', # No mixing for now
        'it contains exactly two blocks.',
        'it contains two or more upright blocks.',
        # 'it contains a block pointing at another block.', # No pointing
        # 'it contains an ungrounded block.', # No grounding for now
        'it contains at least one green block and at least one blue block.',
        # 'it contains at least two blocks that are touching each other.', # No touching for now
    ]

    examples = {
        'it contains at least one red block.': [
            ZendoStructure([ZendoBlock('blue', 'large', 'upright'), 
                            ZendoBlock('blue', 'medium', 'upright')]),
            ZendoStructure([ZendoBlock('blue', 'small', 'upright'), 
                            ZendoBlock('red', 'medium', 'upright'),
                            ZendoBlock('green', 'large', 'flat'),
                            ZendoBlock('yellow', 'small', 'flat')])
        ],
        'it contains at least one small block.': [
            ZendoStructure([ZendoBlock('blue', 'large', 'upright'), 
                            ZendoBlock('blue', 'medium', 'upright')]),
            ZendoStructure([ZendoBlock('blue', 'small', 'upright'), 
                            ZendoBlock('red', 'medium', 'upright'),
                            ZendoBlock('green', 'large', 'flat'),
                            ZendoBlock('yellow', 'small', 'flat')])
        ],
        'it contains at least one block of each of the four colors.': [
            ZendoStructure([ZendoBlock('blue', 'large', 'upright'), 
                            ZendoBlock('blue', 'medium', 'upright')]),
            ZendoStructure([ZendoBlock('blue', 'small', 'upright'), 
                            ZendoBlock('red', 'medium', 'upright'),
                            ZendoBlock('green', 'large', 'flat'),
                            ZendoBlock('yellow', 'small', 'flat')])
        ],
        'it contains no green blocks.': [
            ZendoStructure([ZendoBlock('blue', 'large', 'upright'), 
                            ZendoBlock('green', 'medium', 'upright')]),
            ZendoStructure([ZendoBlock('blue', 'small', 'upright'), 
                            ZendoBlock('red', 'medium', 'upright'),
                            ZendoBlock('yellow', 'small', 'flat')])
        ],
        'it contains no large blocks.': [
            ZendoStructure([ZendoBlock('blue', 'large', 'upright'), 
                            ZendoBlock('green', 'medium', 'upright')]),
            ZendoStructure([ZendoBlock('blue', 'small', 'upright'), 
                            ZendoBlock('red', 'medium', 'upright'),
                            ZendoBlock('yellow', 'small', 'flat')])
        ],
        'it contains exactly two blocks.': [
            ZendoStructure([ZendoBlock('blue', 'small', 'upright'), 
                            ZendoBlock('red', 'medium', 'flat'),
                            ZendoBlock('yellow', 'small', 'flat')]),
            ZendoStructure([ZendoBlock('blue', 'large', 'upright'), 
                            ZendoBlock('green', 'medium', 'upright')]),
        ],
        'it contains two or more upright blocks.': [
            ZendoStructure([ZendoBlock('blue', 'small', 'upright'), 
                            ZendoBlock('red', 'medium', 'flat'),
                            ZendoBlock('yellow', 'small', 'flat')]),
            ZendoStructure([ZendoBlock('blue', 'large', 'upright'), 
                            ZendoBlock('green', 'medium', 'upright')]),
        ],
        'it contains at least one green block and at least one blue block.': [
            ZendoStructure([ZendoBlock('blue', 'small', 'upright'), 
                            ZendoBlock('red', 'medium', 'flat'),
                            ZendoBlock('yellow', 'small', 'flat')]),
            ZendoStructure([ZendoBlock('blue', 'large', 'upright'), 
                            ZendoBlock('green', 'medium', 'upright')]),
        ],
    }
    return simple_rules, examples


def raw_to_stucture(example_raw):
    EPS = np.pi/6
    blocks = []
    contact = list(example_raw['contact'][0].values()) if isinstance(example_raw['contact'][0], dict) else example_raw['contact']
    # TODO Touching is off by 1 -- add 1 please
    for i in range(len(example_raw['colours'])):
        color = example_raw['colours'][i].lower()
        size = ['small', 'medium', 'large'][example_raw['sizes'][i] - 1]
        rot = example_raw['rotations'][i] % (2 * np.pi)
        if abs(rot - np.pi) < EPS:
            orientation = 'upright'
        elif abs(rot - 1.2475) < EPS:
            orientation = 'left'
        elif abs(rot - 5.0375) < EPS:
            orientation = 'right'
        else:
            orientation = 'strange'
        grounded = example_raw['grounded'][i]
        touching = contact[i] if isinstance(contact[i], list) else [i]
        touching.remove(i)

        touching = [x + 1 for x in touching]

        blocks.append(ZendoBlock(color, size, orientation, grounded, touching))
    return ZendoStructure(blocks)

def get_zendo_rules_and_data():
    rules = {
        "zeta": "there's a red block", 
        "phi": "all blocks have the same size",
        "upsilon": "no block is upright",
        "iota": "there is exactly one blue block",
        "kappa": "there's a small blue block",
        "omega": "all blocks are blue or small",
        "mu": "a red block is the largest block",
        "nu": "some blocks are touching",
        "xi": "a blue block touches a red block"
    }

    programs = {
        "zeta": zeta_program, 
        "phi": phi_program,
        "upsilon": upsilon_program,
        "iota": iota_program,
        "kappa": kappa_program,
        "omega": omega_program,
        "mu": mu_program,
        "nu": nu_program,
        "xi": xi_program
    }

    rule_names = ["zeta", "phi", "upsilon", "iota", "kappa", "omega", "mu", "nu", "xi"]

    with open('data/zendo_cases.json', 'r') as f:
        data_list = json.load(f)
    
    return rules, programs, rule_names, data_list

def get_extra_zendo_rules_and_data():
    rules = {
        "more": "more reds than blues", 
        "same": "same number of small and large blocks",
        "even": "even number of blocks oriented right",
    }

    programs = {
        "more": more_program, 
        "same": same_program,
        "even": even_program
    }

    rule_names = ["more", "same", "even"]

    with open('data/zendo_cases_extra.json', 'r') as f:
        data_list = json.load(f)
    
    return rules, programs, rule_names, data_list

def get_adv_zendo_rules_and_examples(extra):
    rules, programs, rule_names, data_list = get_extra_zendo_rules_and_data() if extra else get_zendo_rules_and_data()
    examples = {}
    test_sets = {}
    for rule_name, data in zip(rule_names, data_list):
        examples[rule_name] = [raw_to_stucture(data['t'][0])]
        test_sets[rule_name] = [raw_to_stucture(data['t'][i]) for i in range(1, 5)] + [raw_to_stucture(data['f'][i]) for i in range(1, 5)]

    return rules, rule_names, programs, examples, test_sets

class ZendoModerator:
    def __init__(self, rule, rule_program=None):
        self.rule = rule
        self.rule_program = rule_program
        self.llm = create_llm('gpt-4-1106-preview')

    def query(self, structure):
        if self.rule_program is not None:
            return 'yes' if self.rule_program(structure) else 'no'
        else:
            output = self.llm.prompt([query_prompt.format(rule=self.rule, structure=structure.to_text())],
                                    temperature=0)[0]
            return output.strip(' .\n').lower()
    
    def evaluate_rule(self, rule):
        output = self.llm.prompt([evaluate_rule_prompt.format(rule1=rule, rule2=self.rule)],
                                 temperature=0)[0]
        return output.strip(' .\n').lower()
    

class ZendoBlock:
    def __init__(self, color=None, size=None, orientation=None, groundedness=None, touching=None, txt=None):
        """
        :param color: str
        :param size: str
        :param orientation: str
        :param groundedness: bool
        :param touching: list of int
        :param txt: str
        """
        if (color is None or size is None or orientation is None) and txt is None:
            raise Exception("Everything can't be None")
        
        self.adv_mode = False
        
        if color is not None:
            self.color = color.strip(' .\n').lower()
            self.size = size.strip(' .\n').lower()
            self.orientation = orientation.strip(' .\n').lower()
            self.groundedness = groundedness
            self.touching = touching
        if txt is not None:
            try:
                res = []
                for att in ['Color', 'Size', 'Orientation', 'Groundedness', 'Touching']:
                    if len(txt.split(f'{att} - ')) > 0:
                        res.append(txt.split(f'{att} - ')[1].split(', ')[0].strip(' .\n').lower())
                    elif len(txt.split(f'{att.lower()} - ')) > 0:
                        res.append(txt.split(f'{att.lower()} - ')[1].split(', ')[0].strip(' .\n').lower())
                    else:
                        res.append(None)
                self.color, self.size, self.orientation, self.groundedness, self.touching = res
            except:
                self.color, self.size, self.orientation, self.groundedness, self.touching = [None] * 5
                return

            if self.groundedness is not None:
                self.groundedness = (self.groundedness == 'grounded')
            if self.touching is not None:
                if self.touching.strip(' .\n').lower() == 'none':
                    self.touching = []
                else:
                    try:
                        self.touching = [int(block_str[len('Block '):]) for block_str in self.touching.split(', ')]
                    except:
                        print(f'Cannot parse {self.touching} for touching - will assume it does not touch anything')
                        self.touching = []

        if self.groundedness is not None and self.touching is not None:
            self.adv_mode = True

    def to_text(self, att=None, order=None):
        """
        return str
        """
        if att is not None:
            if att == 'colors':
                return self.color
            elif att == 'sizes':
                return self.size
            elif att == 'orientations':
                return self.orientation
            else:
                if not self.adv_mode:
                    raise Exception(f'attribute {att} not recognized')
                
                if att == 'groundedness':
                    return 'grounded' if self.groundedness else 'ungrounded'
                elif att == 'touchings':
                    # return  ', '.join([f'Block {bid}' for bid in self.touching]) if len(self.touching) > 0 else 'None'
                    return f'{len(self.touching)} Block(s)'
                
                raise Exception(f'attribute {att} not recognized')
        
        if self.adv_mode:
            groundedness_str = 'grounded' if self.groundedness else 'ungrounded'
            touching_str = ', '.join([f'Block {bid}' for bid in self.touching]) if len(self.touching) > 0 else 'None'
            lst = np.asarray([f"Block: Color - {self.color}", f"Size - {self.size}", f"Orientation - {self.orientation}", f"Groundedness - {groundedness_str}", f"Touching - {touching_str}"])
            if order is None:
                order = [0, 1, 2, 3, 4]
            return ", ".join(lst[order])
        else:
            return f"Block: Color - {self.color}, Size - {self.size}, Orientation - {self.orientation}"


class ZendoStructure:
    def __init__(self, blocks=None, txt=None, random_block_rng=None):
        if blocks is None and txt is None and random_block_rng is None:
            raise Exception("Everything can't be None")
        if blocks is not None:
            self.blocks = blocks
        if txt is not None:
            self.blocks = [ZendoBlock(txt=block_txt.replace(' and ', ', ')) for block_txt in txt.strip('\n').split('\n')]
            self.blocks = [block for block in self.blocks if block.color is not None]

            # Ensure touching goes both way
            touchings = {i+1: [] for i in range(len(self.blocks))}
            for idx, block in enumerate(self.blocks):
                for x in block.touching:
                    if x not in touchings:
                        continue
                    touchings[idx + 1].append(x)
                    touchings[x].append(idx + 1)
            for idx, block in enumerate(self.blocks):
                block.touching = sorted(set(touchings[idx + 1]))
        if random_block_rng is not None:
            self.blocks = self.create_random_structure(random_block_rng)
            

    def to_text(self, att=None, order=None):
        if att is not None:
            if att == 'touchings':
                # num_touchings = sum([len(block.touching) for block in self.blocks])
                # return f"Structure: {len(self.blocks)} blocks, touchings = {num_touchings} out of {len(self.blocks) * (len(self.blocks) - 1)}\n"
                # txt = '\n'.join([f'- Block {idx + 1} touches ' + block.to_text(att) for idx, block in enumerate(self.blocks)])
                txt = '\n'.join([f'- A block touches ' + block.to_text(att) for idx, block in enumerate(self.blocks)])
                return f"Structure:\n{txt}\n"
            return f"Structure: {', '.join([block.to_text(att) for block in self.blocks])}\n"
        txt = '\n'.join([f'- Block {idx + 1}: ' + block.to_text(order=order)[len('Block: '):] for idx, block in enumerate(self.blocks)])
        return f"Structure:\n{txt}\n"
    
    def __len__(self):
        return len(self.blocks)
    
    def create_random_structure(self, rng):
        blocks = []
        n_blocks = rng.integers(5) + 1
        for _ in range(n_blocks):
            blocks.append(ZendoBlock(rng.choice(AdvZendoConfig.att_choices_list['colors']),
                                     rng.choice(AdvZendoConfig.att_choices_list['sizes']),
                                     rng.choice(AdvZendoConfig.att_choices_list['orientations']),
                                     rng.choice(AdvZendoConfig.att_choices_list['groundedness']),
                                     []))
            
        # Deal with touching
        possible_touchings = []
        for i in range(1, n_blocks):
            for j in range(i+1, n_blocks):
                possible_touchings.append((i, j))
        n_touchings = rng.geometric(0.3) - 1
        if n_touchings > 0:
            touchings = rng.choice(possible_touchings, size=min(n_touchings, len(possible_touchings)), replace=False)
            for touching in touchings:
                x, y = touching
                blocks[x].touching.append(y + 1)
                blocks[y].touching.append(x + 1)
            for block in blocks:
                block.touching = sorted(block.touching)
        
        return blocks


class ZendoGame:
    def __init__(self, xs, ys, seed=0):
        self.xs = xs
        self.ys = ys
        # self.rng = np.random.default_rng(seed)

        self.good_xs = []
        self.bad_xs = []
        for x, y in zip(self.xs, self.ys):
            if y == 'yes':
                self.good_xs.append(x)
            else:
                self.bad_xs.append(x)

    def to_text(self, att=None, order=None):
        txt = ""
        for idx, x in enumerate(self.good_xs):
            newline_or_not = "" if att is not None else "\n"
            txt += f"Good structure {idx + 1}:{newline_or_not}"
            txt += x.to_text(att, order)[len(f'Structure:{newline_or_not}'):]  + "\n"

        for idx, x in enumerate(self.bad_xs):
            newline_or_not = "" if att is not None else "\n"
            txt += f"Bad structure {idx + 1}:{newline_or_not}"
            txt += x.to_text(att, order)[len(f'Structure:{newline_or_not}'):]  + "\n"

        return txt
    
    def samp_good_bad(self):
        if len(self.bad_xs) == 0:
            return np.random.choice(self.good_xs), None
        return np.random.choice(self.good_xs), np.random.choice(self.bad_xs)

    def samp_two(self):
        xs = self.good_xs + self.bad_xs
        labels = (['yes'] * len(self.good_xs)) + (['no'] * len(self.bad_xs))
        indices = np.random.choice(len(xs), 2)
        if indices[0] == indices[1]:
            return xs[indices[0]], labels[indices[0]], None, None
        return xs[indices[0]], labels[indices[0]], xs[indices[1]], labels[indices[1]]
    
    def append_and_retnew(self, x, y):
        new_game = ZendoGame(list(self.xs) + [x], list(self.ys) + [y])
        return new_game

    def __len__(self):
        return len(self.xs)