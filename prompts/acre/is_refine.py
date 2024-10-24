refine_prompt = """An object contains the following attributes: 
color (gray/red/blue/green/brown/cyan/purple/yellow)
material (metal/rubber)
shape(cube/sphere/cylinder)

Consider the following rule: '{h}'

The given rule gives incorrect output for the following groups of objects:

{feedback}

Based on the given rule, generate {num} new refined rules that fix the outputs for all mentioned structures. 
Please number them from 1-{num} and do not say anything else
"""