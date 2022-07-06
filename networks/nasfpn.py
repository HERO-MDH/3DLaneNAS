"""

Feature Fusion Module

Author: Sadegh Abadijou (s.abadijou@gmail.com)
Date: Nov, 2021

"""

import torch.nn.functional as F
import torch.nn as nn
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class HeadCreator(nn.Module):
    def __init__(self, model_modules, head_id):
        super(HeadCreator, self).__init__()

        self.head_id = head_id

        idx = 0
        for module in model_modules:
            module_name = 'layer'+str(idx)
            self.add_module(module_name, module)
            idx += 1

    def forward(self, x):
        out_layer_1 = list(self.children())[0](x[self.head_id])
        x[self.head_id] = F.relu(out_layer_1)
        for module in list(self.children())[1:]:
            out_module = module(x)
            x[self.head_id] = F.relu(out_module)
        return x[self.head_id]


class ConcatLayer(nn.Module):
    def __init__(self, based_layer, concat_layer, based_id, concat_id, up_sample=True):
        super(ConcatLayer, self).__init__()

        self.based_layer = based_layer

        self.concat_layer = concat_layer

        self.up_sample = up_sample

        self.based_id = based_id

        self.concat_id = concat_id

    def forward(self, x):

        # if concat layer --> x
        # else --> x[id]

        if isinstance(self.based_layer, ConcatLayer):
            out1 = self.based_layer(x)
        else:
            out1 = self.based_layer(x[self.based_id])
        if isinstance(self.concat_layer, ConcatLayer):
            out2 = self.concat_layer(x)
        else:
            out2 = self.concat_layer(x[self.concat_id])

        self.transformerconv = nn.Conv2d(out2.shape[1],
                                         out1.shape[1],
                                         stride=(1, 1),
                                         kernel_size=(1, 1),
                                         device=device)
        out2 = self.transformerconv(out2)

        if self.up_sample:
            scale = int(out1.shape[2] / out2.shape[2])
            out2 = nn.Upsample(scale_factor=scale,
                               mode='nearest')(out2)
            out2 = torch.nn.functional.interpolate(out2, size=(out1.shape[2], out1.shape[3]))
            # Operator #####################################################################
            # out = torch.concat([out1, out2])
            out = out1.add(out2)
            return out
        else:
            scale = int(out2.shape[2] / out1.shape[2])
            out2 = nn.MaxPool2d(kernel_size=scale)(out2)
            # Operator #####################################################################
            out = out1.add(out2)
            # out = torch.concat([out2, out1])

        return out


class NetworkCreator:
    def __init__(self, feature_maps, nas_matrix):
        super(NetworkCreator, self).__init__()

        self.feature_maps = feature_maps

        self.nas_matrix = nas_matrix

        self.layers_head_1 = [self.feature_maps['fm1']]

        self.layers_head_2 = [self.feature_maps['fm2']]

        self.layers_head_3 = [self.feature_maps['fm3']]

        self.layers_head_4 = [self.feature_maps['fm4']]

        self.creator()

    def head_x_1(self):
        head_creator = HeadCreator(self.layers_head_1, 0)
        return head_creator

    def head_x_2(self):
        head_creator = HeadCreator(self.layers_head_2, 1)
        return head_creator

    def head_x_3(self):
        head_creator = HeadCreator(self.layers_head_3, 2)
        return head_creator

    def head_x_4(self):
        head_creator = HeadCreator(self.layers_head_4, 3)
        return head_creator

    def creator(self):

        for column_id in range(0, self.nas_matrix.shape[1]):
            self.layers_head_1 += self.make_layers(layer_exist=self.nas_matrix[0, column_id, 0],
                                                   concat_layer_id=self.nas_matrix[0, column_id, 1],
                                                   layer_row_id=0)

        for column_id in range(0, self.nas_matrix.shape[1]):
            self.layers_head_2 += self.make_layers(layer_exist=self.nas_matrix[0, column_id, 0],
                                                   concat_layer_id=self.nas_matrix[0, column_id, 1],
                                                   layer_row_id=1)

        for column_id in range(0, self.nas_matrix.shape[1]):
            self.layers_head_3 += self.make_layers(layer_exist=self.nas_matrix[0, column_id, 0],
                                                   concat_layer_id=self.nas_matrix[0, column_id, 1],
                                                   layer_row_id=2)

        for column_id in range(0, self.nas_matrix.shape[1]):
            self.layers_head_4 += self.make_layers(layer_exist=self.nas_matrix[0, column_id, 0],
                                                   concat_layer_id=self.nas_matrix[0, column_id, 1],
                                                   layer_row_id=3)

        return True

    def make_layers(self, layer_exist, concat_layer_id, layer_row_id):

        layers_list = [
            self.layers_head_1,
            self.layers_head_2,
            self.layers_head_3,
            self.layers_head_4,
        ]

        if layer_exist == 0:
            return []

        if layer_row_id > concat_layer_id:
            return [ConcatLayer(layers_list[layer_row_id][-1],
                                layers_list[concat_layer_id][-1],
                                based_id=layer_row_id,
                                concat_id=concat_layer_id,
                                up_sample=False)]

        if layer_row_id < concat_layer_id:
            return [ConcatLayer(layers_list[layer_row_id][-1],
                                layers_list[concat_layer_id][-1],
                                based_id=layer_row_id,
                                concat_id=concat_layer_id,
                                up_sample=True)]

        if layer_row_id == concat_layer_id:
            return []


class NasNet(nn.Module):

    def __init__(self, feature_maps_dict, nas_matrix):
        super(NasNet, self).__init__()

        self.network_creator = NetworkCreator(feature_maps_dict, nas_matrix)

        self.head_x1 = self.network_creator.head_x_1().to(device=device)

        self.head_x2 = self.network_creator.head_x_2().to(device=device)

        self.head_x3 = self.network_creator.head_x_3().to(device=device)

        self.head_x4 = self.network_creator.head_x_4().to(device=device)

    def forward(self, input_head_1, input_head_2, input_head_3, input_head_4):
        x = [input_head_1,
             input_head_2,
             input_head_3,
             input_head_4]

        out_1 = self.head_x1(x)
        out_2 = self.head_x2(x)
        out_3 = self.head_x3(x)
        out_4 = self.head_x4(x)

        return out_1, out_2, out_3, out_4
