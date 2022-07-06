"""

State Generator Module for Local Search

Author: 1-  Ali Zoljodi (ali.zoljodi@gmail.com)
        2-  Sadegh Abadijou (s.abadijou@gmail.com)
Date: Nov, 2021

"""

import random
import copy


def gen_one_state(state):
    stage = 1
    if stage == 0:
        mutation_strategy = random.choice(['add_squeeze', 'remove_squeeze', 'increaze_convact', 'reduce_convact'])
        feature = random.choice([0, 1, 2, 3])
        if mutation_strategy == 'add_squeeze':
            try:
                for i in state[0][feature][1]:
                    if i[0] == 'inverted':
                        try:
                            index = random.randint(1, len(i[1]) - 1)
                        except:
                            index = 1
                        if i[1][index][0] == 'convbnact':
                            out = i[1][index][2]
                        elif i[1][index][0] == 'squeeze':

                            out = i[1][index][1]
                        middle = random.choice([16, 32, 64, 128, 256])
                    i[1] = i[1][:index+1]+[['squeeze', out, middle, 'relu']]+i[1][index+1:]
            except Exception as e:
                pass
        elif mutation_strategy == 'increaze_convact':
            layers = []
            for i in range(len(state[0][feature][1])):
                for n in range(len(state[0][feature][1][i][1])):
                    s = state[0][feature][1][i][1][n]
                    if s[0] == 'convbnact':
                        layers.append([feature, i, n, s[1], s[2]])
            try:
                layer = random.choice(layers)
                size = random.choice([32, 64, 128, 256])
                temp = state[0][layer[0]][1][layer[1]][1][layer[2]][2]
                state[0][layer[0]][1][layer[1]][1][layer[2]][2] = size
                state[0][layer[0]][1][layer[1]][1] = state[0][layer[0]][1][layer[1]][1][:layer[2] + 1] + [
                    ['convbnact', size, temp, 'hardswish']] + \
                                                     state[0][layer[0]][1][layer[1]][1][layer[2] + 1:]
            except:
                pass

        elif mutation_strategy == 'reduce_convact':
            layers = []
            for i in state[0][feature][1]:
                for layer in i[1]:
                    layers.append(copy.deepcopy(layer))
            if len(layers)>= 2:
                index = random.randint(0, len(layers) - 2)
                inputS = copy.deepcopy(layers[index][1])
                layers[index + 1][1] = inputS
                layers.remove(layers[index])
                feature_new = [['inverted', layers]]
                state[0][feature][1] = feature_new

        elif mutation_strategy == 'remove_squeeze':
            layers = []
            for i in range(len(state[0][feature][1])):
                for n in range(len(state[0][feature][1][i][1])):
                    s = state[0][feature][1][i][1][n]
                    if s[0] == 'squeeze':
                        layers.append([feature, i, n, s[1], s[2]])
            if len(layers) != 0:
                layer = random.choice(layers)
                state[0][layer[0]][1][layer[1]][1] = state[0][layer[0]][1][layer[1]][1][:layer[2]] + \
                                                          state[0][layer[0]][1][layer[1]][1][layer[2] + 1:]

    elif stage == 1:
        change_row = random.randint(0, 3)
        change_col = random.randint(0, 2)
        change_ex = random.choice([True, False])
        if change_ex:
            if state[1][change_row][change_col][0] == 0:
                input_ = random.randint(0, 3)
                state[1][change_row][change_col][0] = 1
                state[1][change_row][change_col][1] = input_
            else:
                state[1][change_row][change_col][0] = 0
        else:
            input_ = random.randint(0, 3)
            # self.state[1][change_row][change_col][0] = 1
            state[1][change_row][change_col][1] = input_

    return state
