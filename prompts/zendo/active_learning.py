propose_x_prompt = """Given the rule '{h}', please give one structure that conforms with the rule and another structure that violates with the rule. 

A structure has one of more blocks. Each block should contain the following attributes: 
{spec}{stacking_note}

The format of each structure should be as follows:
(conforms with the rule) Structure 1:
{example_block}

(violates the rule) Structure 2:
{example_block}"""

# propose_random_x_prompt = """A structure has one of more blocks. Each block should contain the following attributes: 
# {spec}

# Please generate {n} random structures in the following format:
# Structure x:
# {example_block}

# A structure has {n_blocks} blocks.
# """

propose_llm_x_prompt = """A structure has one of more blocks. Each block should contain the following attributes: 
{spec}{stacking_note}

You are playing a game where you are trying to figure an underlying secret rule governing structures

At this point in the game, you think the underlying secret rule could be any of the following rules:
{hs}

Please choose one structure to ask if it conforms with the underlying secret rule. You want to pick a structure that would help you gain the most information on the secret rule.
The format of the structure should be as follows:
Structure 1:
{example_block}
End of structure 1
"""