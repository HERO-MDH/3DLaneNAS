"""

Random Search Module

Author: Ali Zoljodi (ali.zoljodi@gmail.com)
Date: Nov, 2021

"""

import experiments.test_3DLaneNAS as test_module
import experiments.train_3DLaneNAS as net
import sqlite3
import random
import math


class Random_search():
    def __init__(self,state,dir):
        self.state = state
        self.stage = 1
        self.path = dir
        self.db = dir+'/bests.db'
        self.num = 0
        self.best = math.inf
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute('''CREATE TABLE bestss
                                     (num int, arc text, train_loss real,avg_infer_time real ,energy real,ll_f real, ll_r real, ll_p real
                                     , cl_f real, cl_r real, cl_p real,ll_x_n real,ll_x_f real,ll_z_n real,ll_z_f real,cl_x_n real,cl_x_f real,cl_z_n real,cl_z_f real)''')
        conn.commit()
        c = conn.cursor()
        c.execute('''CREATE TABLE _all_
                                            (num int, arc text, train_loss real,avg_infer_time real, energy real,ll_f real, ll_r real, ll_p real
                                     , cl_f real, cl_r real, cl_p real,ll_x_n real,ll_x_f real,ll_z_n real,ll_z_f real,cl_x_n real,cl_x_f real,cl_z_n real,cl_z_f real)''')
        conn.commit()
        conn.close()


    def run(self):
        self.state=self.move()
        db_entry, model = net.exec(self.state, self.path, self.num, self.past_model, self.change)
        eval_state = test_module.test_3DLane(self.state, self.path, self.num)
        print('lane line')
        print('x error close: ', eval_state[3])
        L_X_close = eval_state[3]
        print('x error far:', eval_state[4])
        L_X_far = eval_state[4]
        print('z error close:', eval_state[5])
        L_Z_close = eval_state[5]
        print('z error far:', eval_state[6])
        L_Z_far = eval_state[6]
        print('center line')
        print('x error close: ', eval_state[10])
        C_X_close = eval_state[10]
        print('x error far:', eval_state[11])
        c_X_far = eval_state[11]
        print('z error close:', eval_state[12])
        C_z_close = eval_state[12]
        print('z error far:', eval_state[13])
        C_Z_far = eval_state[13]
        C_close = math.sqrt(C_X_close ** 2 + C_z_close ** 2)
        L_close = math.sqrt(L_X_close ** 2 + L_Z_close ** 2)
        C_far = math.sqrt(c_X_far ** 2 + C_Z_far ** 2)
        L_far = math.sqrt(L_X_far ** 2 + L_Z_far ** 2)
        Close = (C_close + L_close) / 2
        Far = (C_far + L_far) / 2
        Acc = Close + Far

        loss = db_entry.train_loss
        avg_infer_time = db_entry.latency
        e = Acc * avg_infer_time
        statea = str(self.state)

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute('''INSERT INTO _all_ VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                  [self.num, statea, loss, avg_infer_time, e, db_entry.ll_f_measure, db_entry.ll_recall,
                   db_entry.ll_precision, db_entry.cl_f_measure, db_entry.cl_recal, db_entry.cl_precision,
                   eval_state[3], eval_state[4], eval_state[5],
                   eval_state[6], eval_state[10], eval_state[11], eval_state[12], eval_state[13]])
        conn.commit()
        conn.close()

        if e < self.best:
            conn = sqlite3.connect(self.db)
            c = conn.cursor()
            c.execute('''INSERT INTO bestss VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                      [self.num, statea, loss, avg_infer_time, e, db_entry.ll_f_measure, db_entry.ll_recall,
                       db_entry.ll_precision,
                       db_entry.cl_f_measure, db_entry.cl_recal, db_entry.cl_precision, eval_state[3], eval_state[4],
                       eval_state[5],
                       eval_state[6], eval_state[10], eval_state[11], eval_state[12], eval_state[13]])
            conn.commit()
            conn.close()
            self.best = e
        self.num = self.num + 1

    def move(self):
        if self.stage == 0:
            self.state = 1
            mutation_strategy = random.choice(['add_squeeze', 'remove_squeeze', 'increaze_convact'])
            feature = random.choice([0, 1, 2, 3])
            self.change['F'] = feature

            if mutation_strategy == 'add_squeeze':
                try:
                    for i in self.state[0][feature][1]:
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
                        i[1] = i[1][:index + 1] + [['squeeze', out, middle, 'relu']] + i[1][index + 1:]
                except:
                    pass
            elif mutation_strategy == 'increaze_convact':
                layers = []
                for i in range(len(self.state[0][feature][1])):
                    for n in range(len(self.state[0][feature][1][i][1])):
                        s = self.state[0][feature][1][i][1][n]
                        if s[0] == 'convbnact':
                            layers.append([feature, i, n, s[1], s[2]])
                layer = random.choice(layers)
                size = random.choice([32, 64, 128, 256])
                temp = self.state[0][layer[0]][1][layer[1]][1][layer[2]][2]
                self.state[0][layer[0]][1][layer[1]][1][layer[2]][2] = size
                self.state[0][layer[0]][1][layer[1]][1] = self.state[0][layer[0]][1][layer[1]][1][:layer[2] + 1] + [
                    ['convbnact', size, temp, 'hardswish']] + \
                                                          self.state[0][layer[0]][1][layer[1]][1][layer[2] + 1:]
            elif mutation_strategy == 'reduce_convact':
                layers = []
                for i in range(1, len(self.state[0][feature][1])):
                    for n in range(len(self.state[0][feature][1][i][1])):
                        s = self.state[0][feature][1][i][1][n]
                        if s[0] == 'convbnact':
                            layers.append([feature, i, n, s[1], s[2]])
                try:
                    layer = random.randint(0, len(layers) - 1)
                    layer0 = layers[layer - 1]
                    layer1 = layers[layer]
                    size = layer1[4]
                    self.state[0][layer0[0]][1][layer0[1]][1][layer0[2]][2] = size
                    self.state[0][layer1[0]][1][layer1[1]][1] = self.state[0][layer1[0]][1][layer1[1]][1][:layer1[2]] + \
                                                                self.state[0][layer1[0]][1][layer1[1]][1][
                                                                layer1[2] + 1:]
                except:
                    pass
            elif mutation_strategy == 'remove_squeeze':
                layers = []
                for i in range(len(self.state[0][feature][1])):
                    for n in range(len(self.state[0][feature][1][i][1])):
                        s = self.state[0][feature][1][i][1][n]
                        if s[0] == 'squeeze':
                            layers.append([feature, i, n, s[1], s[2]])
                if len(layers) == 0:
                    print('not squeeze')
                else:
                    layer = random.choice(layers)
                    self.state[0][layer[0]][1][layer[1]][1] = self.state[0][layer[0]][1][layer[1]][1][:layer[2]] + \
                                                              self.state[0][layer[0]][1][layer[1]][1][layer[2] + 1:]
            blocks = []
            for feature in range(4):
                for inverted in range(len(state[feature][1])):
                    for block in range(len(state[feature][1][inverted][1])):
                        blocks.append([feature, inverted, block])

            count = 0
            for i in blocks:
                print(count)
                count += 1
                block = state[i[0]][1][i[1]][1][i[2]]
                if (i[2] < len(state[i[0]][1][i[1]][1]) - 1):
                    next_block = state[i[0]][1][i[1]][1][i[2] + 1]

                else:
                    if (i[1] < len(state[i[0]][1]) - 1):
                        next_block = state[i[0]][1][i[1] + 1][1][0]
                    else:
                        if (i[0] < len(state) - 1):
                            next_block = state[i[0] + 1][1][0][1][0]
                        else:
                            print('list finished')
                            next_block = None
                if next_block is not None:
                    if block[0] == 'convbnact':
                        if block[2] != next_block[1]:
                            next_block[1] = block[2]

                            print('happen')
                        else:
                            print(next_block[1], block[2])

        elif self.stage == 1:
            self.state = 0
            change_row = random.randint(0, 3)
            change_col = random.randint(0, 2)
            change_ex = random.choice([True, False])
            if change_ex:
                if self.state[1][change_row][change_col][0] == 0:
                    input_ = random.randint(0, 3)
                    self.state[1][change_row][change_col][0] = 1
                    self.state[1][change_row][change_col][1] = input_
                else:
                    self.state[1][change_row][change_col][0] = 0
            else:
                input_ = random.randint(0, 3)
                # self.state[1][change_row][change_col][0] = 1
                self.state[1][change_row][change_col][1] = input_

        return self.state


if __name__=='__main__':

    state =[[['F1', [['inverted', [['convbnact', 3, 32, 'hardswish'], ['convbnact', 32, 16, 'hardswish']]]]], ['F2', [['inverted', [['convbnact', 16, 16, 'relu'], ['convbnact', 16, 16, 'identity']]], ['inverted', [['convbnact', 16, 64, 'relu'], ['convbnact', 64, 64, 'relu'], ['convbnact', 64, 24, 'identity']]]]], ['F3', [['inverted', [['convbnact', 24, 72, 'relu'], ['convbnact', 72, 72, 'relu'], ['convbnact', 72, 24, 'identity']]], ['inverted', [['convbnact', 24, 72, 'relu'], ['convbnact', 72, 72, 'relu'], ['convbnact', 72, 40, 'identity']]], ['inverted', [['convbnact', 40, 120, 'relu'], ['convbnact', 120, 32, 'relu'], ['convbnact', 32, 120, 'hardswish'], ['convbnact', 120, 40, 'identity']]], ['inverted', [['convbnact', 40, 120, 'relu'], ['convbnact', 120, 120, 'relu'], ['convbnact', 120, 40, 'identity']]]]], ['F4', [['inverted', [['convbnact', 40, 240, 'hardswish'], ['convbnact', 240, 240, 'hardswish'], ['convbnact', 240, 80, 'identity']]], ['inverted', [['convbnact', 80, 200, 'hardswish'], ['convbnact', 200, 64, 'hardswish'], ['convbnact', 64, 200, 'hardswish'], ['convbnact', 200, 80, 'identity']]], ['inverted', [['convbnact', 80, 184, 'hardswish'], ['convbnact', 184, 80, 'hardswish']]], ['inverted', [['convbnact', 80, 184, 'hardswish'], ['squeeze', 184, 40, 'relu'], ['convbnact', 184, 184, 'hardswish'], ['convbnact', 184, 80, 'identity']]], ['inverted', [['convbnact', 80, 480, 'hardswish'], ['convbnact', 480, 480, 'hardswish'], ['convbnact', 480, 112, 'identity']]], ['inverted', [['convbnact', 112, 672, 'hardswish'], ['convbnact', 672, 672, 'hardswish'], ['convbnact', 672, 112, 'identity']]]]]],[[[0, 2],
         [1, 0],
         [0, 1]],

        [[1, 2],
         [1, 1],
         [1, 1]],

        [[0, 2],
         [0, 0],
         [0, 3]],

        [[1, 3],
         [1, 1],
         [1, 2]]]]

