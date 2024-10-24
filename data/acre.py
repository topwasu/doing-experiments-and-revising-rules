import itertools
import json
import numpy as np


class ACREObject:
    def __init__(self, txt):
        color, material, shape = txt.split(' ')
        self.color = color.strip(' .\n').lower()
        self.material = material.strip(' .\n').lower()
        self.shape = shape.strip(' .\n').lower()

    def to_text(self):
        return f'{self.color} {self.material} {self.shape}'
    

class ACREGroup:
    def __init__(self, objs=None, txt=None, constraints=None):
        if objs is None and txt is None:
            raise Exception("Everything can't be None")
        if objs is not None:
            self.objs = objs
        else:
            try:
                self.objs = [ACREObject(sub_txt) for sub_txt in txt.strip(' .\n').split(', ')]
            except:
                self.objs = []

            if constraints is not None:
                filtered_objs = []
                for obj in self.objs:
                    if obj.to_text() in constraints:
                        filtered_objs.append(obj)
                self.objs = filtered_objs

    def to_text(self):
        return ', '.join([obj.to_text() for obj in self.objs])
    
    def __repr__(self):
        return 'Objs: ' + ', '.join([obj.to_text() for obj in self.objs])
    
    def __len__(self):
        return len(self.objs)
    
    def permute(self, rng):
        return ACREGroup(objs=rng.permutation(self.objs))
    

class ACREGame:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys # on or off

    def to_text(self):
        txt = ""
        for group, y in zip(self.xs, self.ys):
            txt += group.to_text() + f' = light {y}\n'
        return txt
    
    def eval_actual_rule(self, rule):
        all_consistent = True
        for group, y in zip(self.xs, self.ys):
            verdict = 'off'
            for obj in group.objs:
                if rule[obj.to_text()]:
                    verdict = 'on'
                    break
            if verdict != y:
                all_consistent = False
                break
        return all_consistent

    def append_and_retnew(self, group, y):
        new_game = ACREGame(list(self.xs) + [group], list(self.ys) + [y])
        return new_game
    
    def permute(self, rng):
        permuted_xs = [x.permute(rng=np.random.default_rng(rng.integers(10000))) for x in self.xs]
        permuted_xs = np.asarray(permuted_xs)
        ys = np.asarray(self.ys)
        indices = rng.permutation(len(ys))
        return ACREGame(permuted_xs[indices], ys[indices])

    def __len__(self):
        return len(self.xs)


def create_random_group_of_objects():
    colors = ['gray', 'red', 'blue', 'green', 'brown', 'cyan', 'purple', 'yellow']
    materials = ['metal', 'rubber']
    shapes = ['cube', 'sphere', 'cylinder']
    objs_dict = {}
    while len(objs_dict) < 8:
        obj = f'{np.random.choice(colors)} {np.random.choice(materials)} {np.random.choice(shapes)}'
        objs_dict[obj] = True
    return objs_dict


def get_acre_rules_and_examples():
    np.random.seed(0)

    with open('data/acre_data.jsonl', 'r') as json_file:
        json_list = list(json_file)

    data_dicts = []
    for json_str in json_list:
        data_dicts.append(json.loads(json_str))
    
    rules = []
    train_games = []
    test_games = []
    for data_dict in data_dicts:
        # all_obj_names = {}
        # all_latter_obj_names = {}
        # for idx, io in enumerate(data_dict['train']):
        #     for obj_name in io['input']:
        #         all_obj_names[obj_name] = True
        #         if idx > 2:
        #             all_latter_obj_names[obj_name] = True
        all_obj_names = create_random_group_of_objects()
        all_latter_obj_names = all_obj_names

        groups = []
        ys = []
        # for io in data_dict['train'][:3]:
        #     groups.append(ACREGroup([ACREObject(obj_name) for obj_name in io['input']]))
        #     ys.append(io['output'])
        # the last one
        groups.append(ACREGroup([ACREObject(obj_name) for obj_name in all_latter_obj_names]))
        ys.append('on')
        train_game = ACREGame(groups, ys)
        train_games.append(train_game)

        # all_obj_names = all_latter_obj_names # Getting rid of the priming stage
        
        # rule sampling
        rule = dict(zip(list(all_obj_names), np.random.choice([True, False], len(all_obj_names), replace=True)))
        while not train_game.eval_actual_rule(rule): # Have to be consistent with all games
            rule = dict(zip(list(all_obj_names), np.random.choice([True, False], len(all_obj_names), replace=True)))
        rules.append(rule)


        groups = []
        ys = []
        for l in range(1, len(all_obj_names) + 1):
            for subset in itertools.combinations(all_obj_names, l):

                y = 'off'
                groups.append(ACREGroup([ACREObject(obj_name) for obj_name in subset]))
                for obj_name, rule_verdict in rule.items():
                    if obj_name in subset and rule_verdict:
                        y = 'on'
                        break
                ys.append(y)
        
        test_game = ACREGame(groups, ys)
        test_games.append(test_game)

        # groups = []
        # ys = []
        # for io in data_dict['test']:
        #     objs = [ACREObject(obj_name) for obj_name in io['input']]
        #     groups.append(ACREGroup(objs))

        #     # new verdict - in case there's 'undetermined'
        #     new_verdict = 'off'
        #     for obj in objs:
        #         if rule[obj.to_text()]:
        #             new_verdict = 'on'
        #             break
        #     ys.append(new_verdict)
        # test_game = ACREGame(groups, ys)
        # test_games.append(test_game)

    return rules, train_games, test_games
    

class ACREModerator:
    def __init__(self, rule):
        self.rule = rule

    def query(self, group):
        verdict = 'off'
        for obj in group.objs:
            if self.rule[obj.to_text()]:
                verdict = 'on'
                break
        return verdict