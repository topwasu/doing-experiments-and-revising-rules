# new_evolve_h_prompt = """Please make one small modification to the rule '{h}' to take into account the following observation:

# {text_c}

# Give {num} modified rules. The modified rules should still be similar to the original rule. Do not explain.
# """

new_evolve_h_prompt = """An object contains the following attributes: 
color (gray/red/blue/green/brown/cyan/purple/yellow)
material (metal/rubber)
shape(cube/sphere/cylinder)

Example of rule modifications: 
Additional conjunction: 'The light turns on when there is a cylinder present' -> 'The light turns on when there is a cylinder and a cube present'
Additional disjunction: 'The light turns on when there is a cylinder present' -> 'The light turns on when there is a cylinder or a cube present'
Additional attribute: 'The light turns on when there is a cylinder present' -> 'The light turns on when there is a blue cylinder present'
These modifications are "local": only one disjunction/conjunction/attribute is changed or added for each modification.

Please modify the rule '{h}'. Generate {num} rules for each type of modification (Additional conjunction, Additional disjunction, Additional attribute) so that the light does {text_y} turn on when the following objects are present:
{x}
Note that the number of the blocks do not matter.

Make the format a numbered list (1., 2., ..., 15.) Remember that the new rules should be a "local" modification from the rule '{h}'. Do not use attribute values that are not mentioned earlier. Do not say anything other than the modified rules.
"""

basic_propose_h_prompt_w_example = """A group of objects may make the "blicket machine" have lights turned on or off depending on the objects in it. We seek to figure out the rule underlying this. 

Example rules:
'The light turns on when there is a cylinder present'
'The light turns on when there is a cylinder and a cube present'
'The light turns on when there is a cylinder or a cube present'
'The light turns on when there is a blue cylinder present'

Now, consider the following:

{text_c}

Please state {num} possible rules what makes the light turned on. State them in a listed number. Do not explain.
"""

basic_propose_h_prompt = """A group of objects may make the "blicket machine" have lights turned on or off depending on the objects in it. We seek to figure out the rule underlying this. Consider the following:

{text_c}

Please state {num} possible rules what makes the light turned on. State them in a listed number. Do not explain.
"""