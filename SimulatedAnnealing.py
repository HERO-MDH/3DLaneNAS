"""

Simulated Annealing Module

Author: Ali Zoljodi (ali.zoljodi@gmail.com)
Date: Nov, 2021

"""

import experiments.test_3DLaneNAS as test_module
from tools import eval_lane_tusimple, eval_3D_lane
import experiments.train_3DLaneNAS as net
from simanneal import Annealer
from networks import LaneNAS
from tools.utils import *
import sqlite3
import random
import glob
import copy
import math


class SimAnealler(Annealer):
    def __init__(self,state,dir,init_model):
        super(SimAnealler, self).__init__(state)
        self.stage = 1
        self.last_loss = math.inf
        self.eval_state = None
        self.last_e = math.inf
        self.last_avg_time = math.inf
        self.last_arch = None
        self.past_model = init_model
        self.last_db_entry = None
        self.num = 0
        self.best = math.inf
        self.best_v_loss = math.inf
        self.l_f = 0
        self.l_r = 0
        self.l_p = 0
        self.c_f = 0
        self.c_r = 0
        self.c_p = 0
        self.change = {'F': 0,
                       'invertd': 0,
                       'layer': 0}
        # Sqlite Config ###############################################
        self.path = dir
        self.db = dir+'/bests.db'
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

    # Mutate States #####################################################
    def move(self):

        # Backbone Mutation #############################################
        if self.stage == 0:
            self.stage = 1
            mutation_strategy = random.choice(['add_squeeze',
                                               'remove_squeeze',
                                               'increaze_convact',
                                               'reduce_convact'])
            feature = random.choice([0, 1, 2, 3])
            self.change['F'] = feature

            if mutation_strategy == 'add_squeeze':
                try:
                    for i in self.state[0][feature][1]:
                        if i[0] == 'inverted':
                            try:
                                index = random.randint(1, len(i[1])-1)
                            except:
                                index = 1
                            if i[1][index][0] == 'convbnact':
                                out = i[1][index][2]
                            elif i[1][index][0] == 'squeeze':

                                out=i[1][index][1]
                            middle = random.choice([16, 32, 64, 128, 256])
                        i[1]=i[1][:index+1]+[['squeeze', out, middle, 'relu']]+i[1][index+1:]
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
                self.state[0][layer[0]][1][layer[1]][1] = self.state[0][layer[0]][1][layer[1]][1][:layer[2] + 1] + [['convbnact', size, temp,'hardswish']] + \
                                                  self.state[0][layer[0]][1][layer[1]][1][layer[2] + 1:]

            elif mutation_strategy == 'reduce_convact':
                layers = []
                for i in self.state[0][feature][1]:
                    for layer in i[1]:
                        layers.append(copy.deepcopy(layer))
                if len(layers) < 2:
                    print('do nothing')
                else:
                    index = random.randint(0,len(layers)-2)
                    inputS = copy.deepcopy(layers[index][1])
                    layers[index+1][1] = inputS
                    layers.remove(layers[index])
                    feature_new = [['inverted', layers]]
                    self.state[0][feature][1]=feature_new

            elif mutation_strategy=='remove_squeeze':
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
        # Feature Fusion Mutation #############################################
        elif self.stage == 1:
            self.stage = 0
            change_row = random.randint(0,3)
            change_col = random.randint(0,2)
            change_ex = random.choice([True,False])
            if change_ex:
                if self.state[1][change_row][change_col][0] == 0:
                    input_ = random.randint(0, 3)
                    self.state[1][change_row][change_col][0] = 1
                    self.state[1][change_row][change_col][1] = input_
                else:
                    self.state[1][change_row][change_col][0] = 0
            else:
                input_ = random.randint(0, 3)
                self.state[1][change_row][change_col][1] = input_

        return self.energy()

    # Energy Calculation ##########################################################
    def energy(self):
        # Run and Evaluate Architectures ########################################
        if self.state == self.last_arch:
            loss = self.last_loss
            avg_infer_time = self.last_avg_time
            e = self.last_e*1.1
            db_entry = self.last_db_entry
            eval_state = self.eval_state
        else:
            db_entry, model = net.exec(self.state, self.path, self.num, self.past_model, self.change)
            eval_state, avg_time = test_module.test_3DLane(self.state, self.path, self.num)
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
            self.past_model = model
            C_close = math.sqrt(C_X_close ** 2 + C_z_close ** 2)
            L_close = math.sqrt(L_X_close ** 2 + L_Z_close ** 2)
            C_far = math.sqrt(c_X_far ** 2 + C_Z_far ** 2)
            L_far = math.sqrt(L_X_far ** 2 + L_Z_far ** 2)
            Close = (C_close + L_close) / 2
            Far = (C_far + L_far) / 2
            Acc = Close + Far

            loss = db_entry.train_loss
            avg_infer_time = avg_time
            e = Acc * avg_infer_time
        if avg_infer_time > 10000:
            e = e * (avg_infer_time - 10000)

        statea = str(self.state)

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute('''INSERT INTO _all_ VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                  [self.num,statea,loss,avg_infer_time,e,db_entry.ll_f_measure,db_entry.ll_recall,
                  db_entry.ll_precision,db_entry.cl_f_measure,db_entry.cl_recal,db_entry.cl_precision,eval_state[3],eval_state[4],eval_state[5],
                  eval_state[6],eval_state[10],eval_state[11],eval_state[12],eval_state[13]])
        conn.commit()
        conn.close()

        if e < self.best:
            conn = sqlite3.connect(self.db)
            c = conn.cursor()
            c.execute('''INSERT INTO bestss VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                      [self.num, statea, loss,avg_infer_time,e,db_entry.ll_f_measure,db_entry.ll_recall,db_entry.ll_precision,
                      db_entry.cl_f_measure,db_entry.cl_recal,db_entry.cl_precision,eval_state[3],eval_state[4],eval_state[5],
                      eval_state[6],eval_state[10],eval_state[11],eval_state[12],eval_state[13]])
            conn.commit()
            conn.close()
            self.best = e
        self.num = self.num + 1
        import copy
        self.last_model = copy.deepcopy(self.state)
        self.last_db_entry=copy.deepcopy(db_entry)
        self.last_arch=copy.deepcopy(self.state)
        self.last_loss=loss
        self.last_avg_time=avg_infer_time
        self.eval_state=eval_state

        e=math.inf
        return e

# initialize Simulated annealing
def Sim_Annealer(init,init_model):
    path ='/media/nas/DISK1/results/Reduced_new_with_best_starter'
    tsp = SimAnealler(init,path,init_model)
    tsp.Tmax = 25000.0
    tsp.Tmin = 25.0
    tsp.copy_strategy = "deepcopy"
    state, e = tsp.anneal()
    return state


if __name__ == '__main__':

    state = [[['F1', [['inverted', [['convbnact', 3, 32, 'hardswish'], ['convbnact', 32, 16, 'hardswish']]]]], ['F2', [['inverted', [['convbnact', 16, 16, 'relu'], ['convbnact', 16, 16, 'identity']]], ['inverted', [['convbnact', 16, 64, 'relu'], ['convbnact', 64, 64, 'relu'], ['convbnact', 64, 24, 'identity']]]]], ['F3', [['inverted', [['convbnact', 24, 72, 'relu'], ['convbnact', 72, 72, 'relu'], ['convbnact', 72, 24, 'identity']]], ['inverted', [['convbnact', 24, 72, 'relu'], ['convbnact', 72, 72, 'relu'], ['convbnact', 72, 40, 'identity']]], ['inverted', [['convbnact', 40, 120, 'relu'], ['convbnact', 120, 32, 'relu'], ['convbnact', 32, 120, 'hardswish'], ['convbnact', 120, 40, 'identity']]], ['inverted', [['convbnact', 40, 120, 'relu'], ['convbnact', 120, 120, 'relu'], ['convbnact', 120, 40, 'identity']]]]], ['F4', [['inverted', [['convbnact', 40, 240, 'hardswish'], ['convbnact', 240, 240, 'hardswish'], ['convbnact', 240, 80, 'identity']]], ['inverted', [['convbnact', 80, 200, 'hardswish'], ['convbnact', 200, 64, 'hardswish'], ['convbnact', 64, 200, 'hardswish'], ['convbnact', 200, 80, 'identity']]], ['inverted', [['convbnact', 80, 184, 'hardswish'], ['convbnact', 184, 80, 'hardswish']]], ['inverted', [['convbnact', 80, 184, 'hardswish'], ['squeeze', 184, 40, 'relu'], ['convbnact', 184, 184, 'hardswish'], ['convbnact', 184, 80, 'identity']]], ['inverted', [['convbnact', 80, 480, 'hardswish'], ['convbnact', 480, 480, 'hardswish'], ['convbnact', 480, 112, 'identity']]], ['inverted', [['convbnact', 112, 672, 'hardswish'], ['convbnact', 672, 672, 'hardswish'], ['convbnact', 672, 112, 'identity']]]]]], [[[0, 2], [1, 0], [0, 1]], [[1, 2], [1, 1], [1, 1]], [[0, 3], [0, 0], [0, 3]], [[1, 3], [1, 1], [1, 2]]]]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = define_args()

    args = parser.parse_args()
    args.dataset_name = 'illus_chg'
    global evaluator
    if 'tusimple' in args.dataset_name:
        tusimple_config(args)
        # define evaluator
        evaluator = eval_lane_tusimple.LaneEval
    else:
        sim3d_config(args)
        # define evaluator
        evaluator = eval_3D_lane.LaneEval(args)
    args.prob_th = 0.5

    # define the network model
    args.mod = '3D_LaneNet'
    global crit_string
    crit_string = 'loss_3D'

    # for the case only running evaluation
    args.evaluate = False
    args.evaluate = False

    # settings for save and visualize
    args.print_freq = 50
    args.save_freq = 50
    init_model = LaneNAS.Net(args, state=state)
    print(init_model)
    define_init_weights(init_model, args.weight_init)
    if not args.no_cuda:
        # Load model on gpu before passing params to optimizer
        model = init_model.cuda()
    intializer_path = '/media/nas/Disk2/205/illus_chg/3D_LaneNet'
    # load trained model for testing
    best_file_name = glob.glob(os.path.join(intializer_path, 'model_best*'))[0]
    if os.path.isfile(best_file_name):
        sys.stdout = Logger(os.path.join(intializer_path, 'Evaluate.txt'))
        print("=> loading checkpoint '{}'".format(best_file_name))
        checkpoint = torch.load(best_file_name)
        init_model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(best_file_name))

    init_model=None
    SA_state = Sim_Annealer(state,init_model)