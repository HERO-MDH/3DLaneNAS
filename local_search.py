"""

Local Search Module

Author: Sadegh Abadijou (s.abadijou@gmail.com)
Date: Nov, 2021

"""

import experiments.train_3DLaneNAS_localsearch as net
import experiments.test_3DLaneNAS as test_module
from tools.state_gen import gen_one_state
import sqlite3
import torch
import copy
import math
import os

global node_counter
node_counter = 0


def generate_neighbors_state(init_state, parent_state):
    state_list = []
    state = copy.deepcopy(init_state)
    for i in range(50):
        state = gen_one_state(state)
        if (state != init_state) and (state != parent_state):
            if len(state_list) == 0:
                state_list.append(copy.deepcopy(state))
            else:
                s_len = len(state_list)
                for gen_state in state_list:
                    if gen_state != state:
                        s_len -= 1
                if s_len <= 0:
                    state_list.append(copy.deepcopy(state))
    return state_list


class Node:
    def __init__(self, parent, state, freeze_state):
        self.parent = parent
        self.parent_id = -1
        self.state = state
        self.ckpt_dir = None
        self.neighbors = []
        self.neighbors_state = []
        self.freeze_state = freeze_state
        self.accuracy = 0
        self.avg_infer_time = torch.inf
        self.e = torch.inf
        self.loss = torch.inf
        self.node_id = -1
        self.path = '/home/sadegh/PycharmProjects/3DLane_NAS/results'
        self.db_entry = None
        self.eval_state = None
        if self.parent != None:
            self.parent_id = self.parent.node_id

    def train_node(self):
        # Assign node_id ###############
        global node_counter
        self.node_id = node_counter
        node_counter += 1
        # Load_model ###################
        db_entry, model, self.ckpt_dir = net.exec(self.state, self.path, self.node_id, self.parent)
        # Test model ###################
        eval_state, avg_time = test_module.test_3DLane(self.state, self.path, self.node_id)
        # Metrics ######################
        L_X_close = eval_state[3]
        L_X_far = eval_state[4]
        L_Z_close = eval_state[5]
        L_Z_far = eval_state[6]
        C_X_close = eval_state[10]
        c_X_far = eval_state[11]
        C_z_close = eval_state[12]
        C_Z_far = eval_state[13]
        C_close = math.sqrt(C_X_close ** 2 + C_z_close ** 2)
        L_close = math.sqrt(L_X_close ** 2 + L_Z_close ** 2)
        C_far = math.sqrt(c_X_far ** 2 + C_Z_far ** 2)
        L_far = math.sqrt(L_X_far ** 2 + L_Z_far ** 2)
        Close = (C_close + L_close) / 2
        Far = (C_far + L_far) / 2
        self.accuracy = Close + Far
        self.db_entry = db_entry
        self.loss = db_entry.train_loss
        self.avg_infer_time = db_entry.latency
        self.e = self.accuracy * self.avg_infer_time
        self.eval_state = eval_state
        del model

    def gen_neighbors(self):
        print('neighbors generating')
        if self.parent == None:
            self.neighbors_state = generate_neighbors_state(init_state=self.state, parent_state=None)
        else:
            self.neighbors_state = generate_neighbors_state(init_state=self.state, parent_state=self.parent.state)
        print(len(self.neighbors_state))
        i = 0
        for _state in self.neighbors_state:
            i += 1
            node = Node(self, _state, 'freeze_state')
            node.train_node()
            # add to self.neighbors
            self.neighbors.append(node)
        print('neighbor gen done')
        return True


class LocalSearch:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.freeze_state = None
        self.best_e = torch.inf
        self.current_node = None
        self.parent = None

        self.path = r'results_database'
        self.db = self.path + '/bests.db'
        self.create_database()

    def create_database(self):
        try:
            os.remove(self.db)
        except OSError:
            pass

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute('''CREATE TABLE local_search_results
                                             (num int, parent_id int,arc text, train_loss real,avg_infer_time real ,energy real,ll_f real, ll_r real, ll_p real
                                             , cl_f real, cl_r real, cl_p real,ll_x_n real,ll_x_f real,ll_z_n real,ll_z_f real,cl_x_n real,cl_x_f real,cl_z_n real,cl_z_f real, is_best text)''')
        conn.commit()
        conn.close()

    def init(self):
        self.current_node = Node(self.parent, self.initial_state, self.freeze_state)
        self.current_node.train_node()
        self.current_node.gen_neighbors()
        self.best_e = self.current_node.e
        self.add_to_db(self.current_node, 'No')

    def init_middle_nodes(self):
        self.current_node.gen_neighbors()

    def run(self):
        new_e = torch.inf
        # The best node among current neighbors
        new_node = None
        print('Move section')
        for node in self.current_node.neighbors:
            # add new node to db ##########
            self.add_to_db(node)
            ###############################
            if node.e < new_e:
                new_e = node.e
                new_node = node
        if new_e > self.best_e:
            self.add_to_db(self.current_node, 'Yes')
            return self.current_node
        else:
            self.current_node = new_node
            self.freeze_state = self.current_node.freeze_state
            self.initial_state = self.current_node.state
            self.parent = self.current_node.parent
            self.best_e = self.current_node.e
            self.init_middle_nodes()
            # input('run_second_tour')
            self.run()

    def add_to_db(self, node, best_arc='No'):
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute('''INSERT INTO local_search_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                  [node.node_id, node.parent_id, str(node.state), node.db_entry.train_loss, node.avg_infer_time, node.e,
                   node.db_entry.ll_f_measure,
                   node.db_entry.ll_recall,
                   node.db_entry.ll_precision,
                   node.db_entry.cl_f_measure, node.db_entry.cl_recal, node.db_entry.cl_precision,
                   node.eval_state[3], node.eval_state[4], node.eval_state[5],
                   node.eval_state[6], node.eval_state[10], node.eval_state[11], node.eval_state[12],
                   node.eval_state[13],
                   best_arc])

        conn.commit()
        conn.close()


if __name__ == '__main__':
    initial_state = [[['F1', [['inverted', [['convbnact', 3, 32, 'hardswish'], ['convbnact', 32, 16, 'hardswish']]]]],
                      ['F2', [['inverted', [['convbnact', 16, 16, 'relu'], ['convbnact', 16, 16, 'identity']]],
                              ['inverted', [['convbnact', 16, 64, 'relu'], ['convbnact', 64, 64, 'relu'],
                                            ['convbnact', 64, 24, 'identity']]]]], ['F3', [['inverted', [
            ['convbnact', 24, 72, 'relu'], ['convbnact', 72, 72, 'relu'], ['convbnact', 72, 24, 'identity']]],
                                                                                           ['inverted', [
                                                                                               ['convbnact', 24, 72,
                                                                                                'relu'],
                                                                                               ['convbnact', 72, 72,
                                                                                                'relu'],
                                                                                               ['convbnact', 72, 40,
                                                                                                'identity']]],
                                                                                           ['inverted', [
                                                                                               ['convbnact', 40, 120,
                                                                                                'relu'],
                                                                                               ['convbnact', 120, 32,
                                                                                                'relu'],
                                                                                               ['convbnact', 32, 120,
                                                                                                'hardswish'],
                                                                                               ['convbnact', 120, 40,
                                                                                                'identity']]],
                                                                                           ['inverted', [
                                                                                               ['convbnact', 40, 120,
                                                                                                'relu'],
                                                                                               ['convbnact', 120, 120,
                                                                                                'relu'],
                                                                                               ['convbnact', 120, 40,
                                                                                                'identity']]]]], ['F4',
                                                                                                                  [[
                                                                                                                       'inverted',
                                                                                                                       [
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               40,
                                                                                                                               240,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               240,
                                                                                                                               240,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               240,
                                                                                                                               80,
                                                                                                                               'identity']]],
                                                                                                                   [
                                                                                                                       'inverted',
                                                                                                                       [
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               80,
                                                                                                                               200,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               200,
                                                                                                                               64,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               64,
                                                                                                                               200,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               200,
                                                                                                                               80,
                                                                                                                               'identity']]],
                                                                                                                   [
                                                                                                                       'inverted',
                                                                                                                       [
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               80,
                                                                                                                               184,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               184,
                                                                                                                               80,
                                                                                                                               'hardswish']]],
                                                                                                                   [
                                                                                                                       'inverted',
                                                                                                                       [
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               80,
                                                                                                                               184,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'squeeze',
                                                                                                                               184,
                                                                                                                               40,
                                                                                                                               'relu'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               184,
                                                                                                                               184,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               184,
                                                                                                                               80,
                                                                                                                               'identity']]],
                                                                                                                   [
                                                                                                                       'inverted',
                                                                                                                       [
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               80,
                                                                                                                               480,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               480,
                                                                                                                               480,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               480,
                                                                                                                               112,
                                                                                                                               'identity']]],
                                                                                                                   [
                                                                                                                       'inverted',
                                                                                                                       [
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               112,
                                                                                                                               672,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               672,
                                                                                                                               672,
                                                                                                                               'hardswish'],
                                                                                                                           [
                                                                                                                               'convbnact',
                                                                                                                               672,
                                                                                                                               112,
                                                                                                                               'identity']]]]]],
                     [[[0, 2],
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

    approach = LocalSearch(initial_state=initial_state)
    approach.init()
    best_node = approach.run()
    print('***...................FINISH'
          '.....................***')
