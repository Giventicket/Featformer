import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import os
from tqdm import tqdm
import math


class GroupState:
    def __init__(self, group_size, x):
        # x.shape = [B, N, 2]
        self.batch_size = x.size(0)
        self.group_size = group_size
        self.device = x.device

        self.selected_count = 0
        # current_node.shape = [B, G]
        self.current_node = None
        # selected_node_list.shape = [B, G, selected_count]
        self.selected_node_list = torch.zeros(x.size(0), group_size, 0, device=x.device).long()
        # ninf_mask.shape = [B, G, N]
        self.ninf_mask = torch.zeros(x.size(0), group_size, x.size(1), device=x.device)

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = [B, G]
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)
        self.ninf_mask.scatter_(dim=-1, index=selected_idx_mat[:, :, None], value=-torch.inf)


class MultiTrajectoryTSP:
    def __init__(self, x, x_raw=None, integer=False):
        self.integer = integer
        if x_raw is None:
            self.x_raw = x.clone()
        else:
            self.x_raw = x_raw
        self.x = x
        self.batch_size = self.B = x.size(0)
        self.graph_size = self.N = x.size(1)
        self.node_dim = self.C = x.size(2)
        self.group_size = self.G = None
        self.group_state = None

    def reset(self, group_size):
        self.group_size = group_size
        self.group_state = GroupState(group_size=group_size, x=self.x)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = self.group_state.selected_count == self.graph_size
        if done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_group_travel_distance(self):
        # ordered_seq.shape = [B, G, N, C]
        shp = (self.B, self.group_size, self.N, self.C)
        gathering_index = self.group_state.selected_node_list.unsqueeze(3).expand(*shp)
        seq_expanded = self.x_raw[:, None, :, :].expand(*shp)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        # segment_lengths.size = [B, G, N]
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        if self.integer:
            group_travel_distances = segment_lengths.round().sum(2)
        else:
            group_travel_distances = segment_lengths.sum(2)
        return group_travel_distances


def readDataFile(filePath):
    """
    read validation dataset from "https://github.com/Spider-scnu/TSP"
    """
    res = []
    with open(filePath, "r") as fp:
        datas = fp.readlines()
        for data in datas:
            data = [float(i) for i in data.split("o")[0].split()]
            loc_x = torch.FloatTensor(data[::2])
            loc_y = torch.FloatTensor(data[1::2])
            data = torch.stack([loc_x, loc_y], dim=1)
            res.append(data)
    res = torch.stack(res, dim=0)
    return res


def readTSPLib(filePath):
    """
    read TSPLib
    """
    data_trans, data_raw = [], []
    with open(filePath, "r") as fp:
        loc_x = []
        loc_y = []
        datas = fp.readlines()
        for data in datas:
            if ":" in data or "EOF" in data or "NODE_COORD_SECTION" in data:
                continue
            data = [float(i) for i in data.split()]
            if len(data) == 3:
                loc_x.append(data[1])
                loc_y.append(data[2])
        loc_x = torch.FloatTensor(loc_x)
        loc_y = torch.FloatTensor(loc_y)

        data = torch.stack([loc_x, loc_y], dim=1)
        data_raw.append(data)

        mx = loc_x.max() - loc_x.min()
        my = loc_y.max() - loc_y.min()
        data = torch.stack([loc_x - loc_x.min(), loc_y - loc_y.min()], dim=1)
        data = data / max(mx, my)
        data_trans.append(data)

    data_trans = torch.stack(data_trans, dim=0)
    data_raw = torch.stack(data_raw, dim=0)
    return data_trans, data_raw


def readTSPLibOpt(opt_path):
    with open(opt_path, "r") as fp:
        datas = fp.readlines()
        tours = []
        for data in datas:
            if ":" in data or "-1" in data or "TOUR_SECTION" in data or "EOF" in data:
                continue
            tours.extend([int(i) - 1 for i in data.split()])
        tours = np.array(tours, dtype=np.int)
    return tours


class TSPDataset(Dataset):
    def __init__(
        self,
        size=50,
        node_dim=2,
        num_samples=100000,
        data_distribution="uniform",
        data_path=None,
        embedding_dim=128,
        use_position_encoding=True,
        position_encoding_type="xy_sum",  # xy_concat, xy_sum, rad_sum, rad_concat, discretized_{}
    ):
        super(TSPDataset, self).__init__()

        print()
        print("#### creating dataset... ####")
        print("data_path:", data_path)
        print("num_samples:", num_samples)
        print("data_distribution:", data_distribution)
        print("embedding_dim:", embedding_dim)
        print("position_encoding_type:", position_encoding_type)
        print("#############################")
        print()

        if data_distribution == "uniform":
            # print("data generated from uniform")
            self.data = torch.rand(num_samples, size, node_dim)
        elif data_distribution == "normal":
            # print("data generated from normal")
            self.data = torch.randn(num_samples, size, node_dim)
        self.size = num_samples
        if not data_path is None:
            # print("data generated from path")
            # print(data_path)
            if data_path.split(".")[-1] == "tsp":
                self.data, data_raw = readTSPLib(data_path)
                opt_path = data_path.replace(".tsp", ".opt.tour")
                print("opt_path", opt_path)
                if os.path.exists(opt_path):
                    self.opt_route = readTSPLibOpt(opt_path)
                    tmp = np.roll(self.opt_route, -1)
                    d = data_raw[0, self.opt_route] - data_raw[0, tmp]
                    self.opt = np.linalg.norm(d, axis=-1).sum()
                else:
                    self.opt = -1
                self.data = data_raw

            else:
                self.data = readDataFile(data_path)
            self.size = self.data.shape[0]

        # self.data = self.data[:100] # for fast debugging

        self.use_position_encoding = use_position_encoding
        if use_position_encoding:
            self.angle_rates = None

            if "discretized" in position_encoding_type:
                increment = 0.001
                if "xy" in position_encoding_type:
                    self.sin_table = torch.sin(torch.linspace(0, 1, int(1 / increment)))
                    self.cos_table = torch.cos(torch.linspace(0, 1, int(1 / increment)))
                    self.norm_weight = 1

                if "rad" in position_encoding_type:
                    self.sin_table = torch.sin(torch.linspace(0, math.pi / 2, int((math.pi / 2) / increment)))
                    self.cos_table = torch.cos(torch.linspace(0, math.pi / 2, int((math.pi / 2) / increment)))
                    self.norm_weight = math.pi / 2

            self.embedding_dim = embedding_dim
            self.position_encoding_type = position_encoding_type

            self.pe_caches = []
            for datum in tqdm(self.data):
                pe_cache = []
                for j in range(size):
                    xx = datum[j, 0]
                    yy = datum[j, 1]
                    pe_cache.append(self.positional_encoding(xx, yy))
                pe_cache = torch.stack(pe_cache)
                self.pe_caches.append(pe_cache)

    def positional_encoding(self, x, y):
        d_model = self.embedding_dim
        pe_type = self.position_encoding_type

        assert d_model % 2 == 0
        if "concat" in pe_type:
            d_model = d_model // 2

        if "rad" in pe_type:
            if not isinstance(self.angle_rates, torch.Tensor):
                T = math.pi / 2
                self.angle_rates = 1 / torch.pow(T, (2.0 * (torch.arange(d_model) // 2)) / torch.tensor(d_model))
            angle_rads = torch.atan2(y, x) * self.angle_rates
            angle_rads2 = torch.sqrt(x**2 + y**2) * self.angle_rates

        elif "xy" in pe_type:
            if not isinstance(self.angle_rates, torch.Tensor):
                # T = math.sqrt(2)
                T = 2
                self.angle_rates = 1 / torch.pow(T, (2.0 * (torch.arange(d_model) // 2)) / torch.tensor(d_model))
            angle_rads = x * self.angle_rates
            angle_rads2 = y * self.angle_rates
        else:
            raise ValueError("Wrong input for mode")

        if "discretized" in pe_type:
            angle_rads[::2] = self.sin_table[(angle_rads[::2] / self.norm_weight * len(self.sin_table)).long()]
            angle_rads[1::2] = self.cos_table[(angle_rads[1::2] / self.norm_weight * len(self.cos_table)).long()]
            angle_rads2[::2] = self.sin_table[(angle_rads2[::2] / self.norm_weight * len(self.sin_table)).long()]
            angle_rads2[1::2] = self.cos_table[(angle_rads2[1::2] / self.norm_weight * len(self.cos_table)).long()]
        else:
            angle_rads[::2] = torch.sin(angle_rads[::2])
            angle_rads[1::2] = torch.cos(angle_rads[1::2])
            angle_rads2[::2] = torch.sin(angle_rads2[::2])
            angle_rads2[1::2] = torch.cos(angle_rads2[1::2])

        if "concat" in pe_type:
            return torch.cat([angle_rads, angle_rads2])

        elif "sum" in pe_type:
            return angle_rads + angle_rads2

        else:
            raise ValueError("Wrong input for mode")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # self.data[idx].shape = [100, 2]
        if self.use_position_encoding:
            pe = self.pe_caches[idx]
            return [self.data[idx], pe]
        else:
            return self.data[idx]


if __name__ == "__main__":
    TSPDataset(data_path="./data/ALL_tsp/att48.tsp")
