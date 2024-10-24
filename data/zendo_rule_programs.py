def zeta_program(structure):
    for block in structure.blocks:
        if block.color == 'red':
            return True
    return False

def phi_program(structure):
    if len(structure.blocks) == 0:
        return False
    target_size = structure.blocks[0].size
    for block in structure.blocks:
        if block.size != target_size:
            return False
    return True

def upsilon_program(structure):
    for block in structure.blocks:
        if block.orientation == 'upright':
            return False
    return True

def iota_program(structure):
    blue_ct = 0
    for block in structure.blocks:
        if block.color == 'blue':
            blue_ct += 1
    if blue_ct == 1:
        return True
    return False

def kappa_program(structure):
    for block in structure.blocks:
        if block.color == 'blue' and block.size == 'small':
            return True
    return False

def omega_program(structure):
    for block in structure.blocks:
        if block.color != 'blue' and block.size != 'small':
            return False
    return True

def mu_program(structure):
    biggest_size = 'small'
    for block in structure.blocks:
        if block.size == 'large':
            biggest_size = 'large'
        elif block.size == 'medium' and biggest_size == 'small':
            biggest_size = 'medium'
    for block in structure.blocks:
        if block.size == biggest_size and block.color != 'red':
            return False
    return True

def nu_program(structure):
    for block in structure.blocks:
        if len(block.touching) != 0:
            return True
    return False

def xi_program(structure):
    for block in structure.blocks:
        if block.color == 'blue' and len(block.touching) != 0:
            for touch in block.touching:
                if structure.blocks[touch - 1].color == 'red':
                    return True
    return False

def psi_program(structure):
    for block in structure.blocks:
        if block.stacking is not None:
            return True
    return False

def more_program(structure):
    red_ct, blue_ct = 0, 0
    for block in structure.blocks:
        if block.color == 'red':
            red_ct += 1
        elif block.color == 'blue':
            blue_ct += 1
    if red_ct > blue_ct:
        return True
    return False

def same_program(structure):
    small_ct, large_ct = 0, 0
    for block in structure.blocks:
        if block.size == 'small':
            small_ct += 1
        elif block.size == 'large':
            large_ct += 1
    if small_ct == large_ct:
        return True
    return False

def even_program(structure):
    right_ct = 0
    for block in structure.blocks:
        if block.orientation == 'right':
            right_ct += 1
    if right_ct % 2 == 0:
        return True
    return False

def red_program(structure):
    red_ct = 0
    total_blocks = len(structure.blocks)
    for block in structure.blocks:
        if block.color == 'red':
            red_ct += 1
    return red_ct > total_blocks / 2