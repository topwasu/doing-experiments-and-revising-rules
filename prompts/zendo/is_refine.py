propose_h_prompt = """Suppose each block can have {att} {att_choices}.
Given the following structure described with {att} of blocks in the structure:
{text_c}
Please list {num} possible rules about the {att} in a structure that differentiate the good structures from the bad structures.
Keep in mind that 
1. All bad structures must violate the rules.
2. Orders of blocks in a structure do NOT matter. Do not include rules about the position of the {att}
Please number them from 1-{num} and do not say anything else
"""

propose_h_all_prompt = """Given the following structures described with {att_summary} of blocks in the structures:
{text_c}
Please list {num} possible rules about the attributes in a structure that differentiate the good structures from the bad structures.
Keep in mind that 
1. All bad structures must violate the rules.
2. Orders of blocks in a structure do NOT matter.
3. The rules are short, concise, single sentences.
Please number them from 1-{num} and do not say anything else
"""

propose_h_all_no_neg_prompt = """Given the following structures described with {att_summary} of blocks in the structures:
{text_c}
Please list {num} possible rules about the attributes in a structure that differentiate the good structures from the bad structures.
Keep in mind that 
1. All bad structures must violate the rules.
2. Orders of blocks in a structure do NOT matter.
3. Do NOT propose "negative rules" such as "there is no green block".
4. The rules are short, concise, single sentences.
Please number them from 1-{num} and do not say anything else
"""

refine_prompt = """A structure has one or more blocks. Each block should contain the following attributes: 
{att_par}

Consider the following rule: '{h}'

Given a structure, the output is yes if it follows the rule (or is a good structure) and no if it does not (or is a bad structure)

The given rule gives incorrect output for the following structures:

{feedback}

Based on the given rule, generate {num} new refined rules that fix the outputs for all mentioned structures. The new rules may involve any of the mentioned attributes (color, size, orientation, grounded, touching).
Please number them from 1-{num} and do not say anything else
"""