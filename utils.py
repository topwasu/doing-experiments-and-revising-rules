def parse_listed_output(outputs):
    # TODO: make more robust
    try:
        # return [output.split('. ')[1].strip('\n') for output in list(filter(None, outputs.split('\n')))]
        idx = 1
        res = []
        for output in list(filter(None, outputs.split('\n'))):
            if len(output.split(f'{idx}. ')) > 1:
                res.append(output.split(f'{idx}. ')[1].strip(' \n'))
                idx += 1
        return res
    except:
        print("SOMETHING WRONG", outputs, list(filter(None, outputs.split('\n'))))


def list_to_str(lst):
    res = ''
    for idx, x in enumerate(lst): 
        res += f'{idx + 1}. {x}\n'
    return res