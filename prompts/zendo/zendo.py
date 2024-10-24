query_prompt = """Given the rule {rule}, please give a yes or no answer on whether the following structure conform with the rule:
{structure}
If yes, say 'yes'. If no, explain why not."""


evaluate_rule_prompt = """Are these two rules the same rule?
{rule1}
vs
{rule2}
If yes, say 'yes'. If no, give an example that conforms with the first rule but not the second rule."""

play_zendo_easy_prompt = """You are playing an inductive game with me. I'll be the moderator, and your task is to figure out the secret rule determining whether a structure of blocks is good or bad. 
You will do that by coming up with a structure of blocks and asking me whether it is a good structure according to the rule. 

The structure has one of more blocks. Each block should contain the following attributes: 
color (blue/red/green/yellow) 
size (small/medium/large)
orientation (upright/flat)

To give you a start, I'll describe one structure that follows the rule:

{text_c}"""

play_zendo_hard_prompt = """You are playing an inductive game with me. I'll be the moderator, and your task is to figure out the secret rule that I know by coming up with a structure of blocks to ask me whether it conforms with the secret rule or not. 

The structure has one of more blocks. Each block should contain the following attributes: 
color (blue/red/green) 
size (small/medium/large)
orientation (upright/left/right/strange)
groundedness (grounded/ungrounded),
touching (which other blocks they do and do not touch).

To give you a start, I'll describe one structure that follows the rule:

{text_c}

Give a very short summary on what you currently think the secret rule is."""

x_conforms_h_prompt = """Given the rule about good structure '{h}', is the following structure a good structure?

{x} 

Note that right and upright orientations are different. Say 'yes' or 'no'. Do not say anything else."""

commonalities_prompt = """Please summarize the commonalities among the good structures and the bad structures
{text_c}"""

rule_translation_prompt = """Please synthesize a python program that implements the rule '{h}'

The program should takes in a ZendoStructure which represents a structure and returns True if it's a good structure and False otherwise.

The docstrings for the classes are as follow:

class ZendoStructure:
    :param blocks: list of ZendoBlock

class ZendoBlock:
    :param color: str (blue/red/green) 
    :param size: str (small/medium/large)
    :param orientation: str (upright/left/right/strange)
    :param groundedness: bool
    :param touching: list of int (index starts at 1)

The signature for the synthesized program should be
def rule(structure: ZendoStructure) -> bool

Only output the 'rule' function. Do not include anything else.
"""
