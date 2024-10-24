play_prompt = """You are playing an inductive game with me. I'll be the moderator, and your task is to figure out the secret rule that I know by coming up with a group of blocks to ask me whether the group conforms with the secret rule or not. 

An object contains the following attributes: 
color (gray/red/blue/green/brown/cyan/purple/yellow)
material (metal/rubber)
shape(cube/sphere/cylinder)
The list of available of objects are {all_objects}.

To give you a start, I'll describe one group of objects that follows the rule:

{text_c}

Give a very short summary on what you currently think the secret rule is."""

rule_translation_prompt = """Please synthesize a python program that implements the rule '{h}'

The program should takes in a ACREGroup which represents a group of objects and returns True if it's a good group and False otherwise.

The docstrings for the classes are as follow:

class ACREGroup:
    :param objs: list of ACREObject

class ACREObject:
    :param color: str (gray, red, blue, green, brown, cyan, purple, yellow)
    :param material: str (metal, rubber)
    :param shape: str (cube, sphere, cylinder)

The signature for the synthesized program should be
def rule(group: ACREGroup) -> bool

Only output the 'rule' function. Do not include anything else.
"""

propose_x_prompt = """Given the rule '{h}', please give one group of objects that makes the light turned on and another that makes the light turned off

The list of available of objects are {all_objects}.

The format of your answer should as follows:

light on group of objects: obj_1, obj_2, ...

light off group of objects: obj_1, obj_2, ...

All objects in a group must be unique. Do not say anything else."""