evolve_h_prompt = """A structure has one or more blocks. Each block should contain the following attributes: 
color (blue/red/green) 
size (small/medium/large)
orientation (upright/left/right/strange)
groundedness (grounded/ungrounded),
touching (which other blocks they do and do not touch).

Example of rule modifications: 
{rule_examples}

Please modify the rule '{h}'. Generate {num} rules with '{mode}' modification so that the following structure is {text_y} a good structure:
{x}

Make the format a numbered list (1., 2., etc.). Remember that the new rules should be a small modification from the rule '{h}'. Do not say anything other than the modified rules.
"""

evolve_h_modes = ['quantifier', 'additional attribute', 'change attribute']

evolve_h_mode_examples = [
"""Quantifier: 'There must be a green block' -> 'There must be exactly one green blocks'
Quantifier: 'There must be at least three green block' -> 'There must at least one green block'
Quantifier: 'There must be a green block' -> 'There must be a green block and a blue block'
""",
"""Additional attribute: 'There must be a green block' -> 'There must be a green block that is upright'
""",
"""Change attribute: 'There must be a green block that is upright' -> 'There must be a green block that is grounded'
""",
]

new_evolve_h_prompt = """A structure has one or more blocks. Each block should contain the following attributes: 
{att_par}

Example of rule modifications: 
Quantifier change: 'There must be a green block' -> 'There are two green blocks'
Additional attribute: 'There must be a green block' -> 'There must be a green block that is upright'
Attribute change: 'There must be a green block' -> 'There must be a blue block'
These modifications are "local": only one attribute/quantifier is changed or added for each modification.

Please modify the rule '{h}'. Generate {num} rules for each type of modification (Quantifier change, Additional attribute, Attribute change) so that the following structure is {text_y} a good structure:
{x}
Note that the number of the blocks do not matter.

Make the format a numbered list (1., 2., ..., 15.) Remember that the new rules should be a "local" modification from the rule '{h}'. Do not use attribute values that are not mentioned earlier. Do not say anything other than the modified rules.
"""

basic_propose_h_prompt = """Please list {num} possible rules about the {att}.

Example 1:
Structure: blue, blue
Simple rules (Orders do NOT matter):
1. There is a blue block
2. All blocks are blue
Do NOT propose "negative rules" such as "there is no green block". Do NOT propose rules with quantifier such as "there are two blue blocks"

Task 1:
{x}
Simple rules (Orders do NOT matter):
"""

propose_h_all_basic_prompt = """Given the following structures described with {att_summary} of blocks in the structures:
{text_c}
Please list {num} possible rules about the attributes in a structure that differentiate the good structures from the bad structures.
Keep in mind that 
1. All bad structures must violate the rules.
2. Orders of blocks in a structure do NOT matter.
3. The rules are short, concise, single sentences.
4. The rules are very simple.
Please number them from 1-{num} and do not say anything else
"""

prior_prompt = """Your task is to list the attribute instances involved in the given rule about blocks.
Example:
'There are exactly three blocks.': [three (quantity)]
'There is no grounded blocks': [no/zero (quantity), grounded (groundedness)]
'There are at least two small blue blocks.': [two (quantity), small (size), blue (color)]
'A blue block touches a red block.' : [blue (color), red (color), touching (action)] 
Task:
'{h}':
Give your answer without explanation"""