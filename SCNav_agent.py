from models import QNet, ResNet, Bottleneck,\
DeconvBottleneck, ACNet, Q_discrete
from models.ResNet import ResNet_dist
from models.QNet import State_predictor
#from models import QNet, QNet512, SegNet, ConfNet, ResNet, Bottleneck,\
#DeconvBottleneck, ACNet, Q_discrete
#from pathfinding.core.diagonal_movement import DiagonalMovement
#from pathfinding.core.grid import Grid
#from pathfinding.finder.a_star import AStarFinder
from utils.dataloader_seg import *

import habitat
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import (
            ObjectGoal,
                ObjectGoalNavEpisode,
                    ObjectViewLocation,
                    )

from einops import asnumpy
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision

import pickle
import _pickle as cPickle

import random
import numpy as np
#import pandas as pd
from quaternion import as_rotation_matrix, from_rotation_matrix
from utils.utils import generate_pc, color2local3d, repeat4, pc2local, pc2local_gpu, d3_41_colors_rgb
from collections import namedtuple, OrderedDict
import copy
import os
from utils.mapper import Mapper
from utils.module_relation_tracking import RelationTrackingModule
#import skfmm
name2id = {
        'door': 3,
        'table': 4,
        'sofa': 9,
        'bed': 10,
        'sink': 14,
        'toilet': 17,
        'bathtub': 24,
        'shower': 22,
        'counter': 25
        }

classes_SSCNav = ['wall', 'floor', 'chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa',
                  'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs', 'ceiling', 'toilet', 'stool',
                  'towel', 'mirror', 'tv_monitor', 'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting',
                  'beam', 'railing', 'shelving', 'blinds', 'gym_equipment', 'seating', 'board_panel', 'furniture',
                  'appliances', 'clothes', 'objects', 'misc']

classes_PONI = ["floor", "wall", "chair", "table", "picture", "cabinet", "cushion", "sofa", "bed", "chest_of_drawers",
                "plant", "sink", "toilet", "stool", "towel", "tv_monitor", "shower", "bathtub", "counter", "fireplace",
                "gym_equipment", "seating", "clothes"]

idxs = []
for poni in classes_PONI:
    if poni in classes_SSCNav:
        idxs.append(classes_SSCNav.index(poni))

layer_infos = [
                  [64, 7, 2, 3],
                  [3, 2, 1],
                  [12, 64, 3, 2, 1, 1],
                  [16, 128, 3, 1, 1, 1],
                  [24, 256, 3, 1, 1, 1],
                  [12, 512, 3, 1, 1, 1],
                  [12, 512, 3, 1, 1, 0, 1],
                  [24, 256, 3, 1, 1, 0, 1],
                  [16, 128, 3, 1, 1, 0, 1],
                  [12, 64, 3, 2, 1, 1, 1],
                  [4, 64, 3, 2, 1, 1, 1]
    ]

transform = transforms.Compose([
    scaleNorm(),
    ToTensor(),
    Normalize()
    ])

Transition = namedtuple('Transition', ('state', 'relationship_bank', 'action', 'next_state',
                                       'next_relationship_bank', 'reward'))

class replay_buffer(object):
    def __init__(self, capacity, save_dir, current_position):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        if current_position is None:
            self.save_dir = os.path.join(save_dir, "replay_buffer")
            if os.path.exists(self.save_dir):
                assert False, "Replay Buffer Directory already exists!"
            os.mkdir(self.save_dir)
        else:
            self.save_dir = os.path.join(save_dir, 'replay_buffer')
            self.position = current_position
            files = [f for f in os.listdir(self.save_dir) if f.endswith('pkl')]
            for fid in range(len(files)):
                f = '.'.join([str(fid), 'pkl'])
                self.memory.append(os.path.join(self.save_dir, f))


    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        target_file = os.path.join(self.save_dir, "%s.pkl" % (self.position))
        with open(target_file, 'wb') as f:
            cPickle.dump(Transition(*args), f, protocol=-1)
        self.memory[self.position] = target_file
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):

        paths = random.sample(self.memory, batch_size)
        result = []
        for path in paths:
            with open(path, 'rb') as f:
                data = cPickle.load(f)
                result.append(data)

        return result


    def __len__(self):
        return len(self.memory)


class Memory:
    def __init__(
            self, 
            aggregate, 
            memory_size, 
            pano,
            num_channel, 
            id2cat, 
            roof_thre,
            floor_thre, 
            ignore, 
            fov=np.pi / 2.):
        self.points = None
        self.rgbs = None
        self.semantics = None
        
        self.aggregate = aggregate
        self.pano = pano
        self.memory_size = memory_size
        self.markers = []

        self.num_channel = num_channel
        self.id2cat = id2cat
        self.roof_thre = roof_thre
        self.floor_thre = floor_thre
        self.ignore = ignore
        self.fov = fov
        
        self.ready_counter = 0

    def reset(self):
        self.points = None
        self.rgbs = None
        self.semantics = None
        self.markers = []
        self.ready_counter = 0

    def get_height_map(self, quaternion, translation, area_x, area_z, h, w):
        if self.pano:
            assert self.ready_counter ==0, "memory accumulating for pano case"
        assert len(self.markers) > 0, "Cannot view an empty memory"
        
        rotation = as_rotation_matrix(quaternion)
        T_world = np.eye(4)
        T_world[0:3, 0:3] = rotation
        T_world[0:3, 3] = translation

        pointss = pc2local(T_world, self.points)
        round_agent = (np.abs(pointss[:, 0]) < area_x / 2.)\
                    & (np.abs(pointss[:, 2]) < area_z / 2.)
        pointss = pointss[round_agent, :]
        rgbss = self.rgbs[round_agent, :]
        semanticss = self.semantics[round_agent]

        scale_h = area_z / h
        scale_w = area_x / w
        X = (pointss[:, 0] + (area_x / 2.)) / float(scale_w)
        Y = pointss[:, 1]
        Z = (pointss[:, 2] + (area_z / 2.)) / float(scale_h)
        
        # 1 for 4
        XZ_ff = np.column_stack((np.floor(X), np.floor(Z))) #0, 1, ... n-1
        XZ_fc = np.column_stack((np.floor(X), np.ceil(Z)))
        XZ_cc = np.column_stack((np.ceil(X), np.ceil(Z)))
        XZ_cf = np.column_stack((np.ceil(X), np.floor(Z)))
        XZ = np.concatenate((np.concatenate((XZ_ff, XZ_fc), axis=0),\
                np.concatenate((XZ_cc, XZ_cf), axis=0)), axis=0) #0, 1, ... 4n - 1
        assert h == w, "Currently only support square!"
        XZ[XZ >= h] = h - 1
        XZ[XZ < 0.] = 0.
        Y = repeat4(Y)
        
        XYZ = np.column_stack((XZ, Y))
        df = pd.DataFrame(XYZ)
        idx = df.groupby([0, 1])[2].transform(max) == df[2]

        idx = idx.values

        height_rgb = np.zeros((h, w, 3))
        height_sem = np.ones((h, w), dtype=np.int32) * (self.num_channel - 1)
        rgbss_4 = repeat4(rgbss)
        semanticss_4 = repeat4(semanticss)

        XZ = XZ.astype(int)
        height_rgb[XZ[idx][:, 0], XZ[idx][:, 1], :] = rgbss_4[idx, :]
        height_sem[XZ[idx][:, 0], XZ[idx][:, 1]] = semanticss_4[idx]
        height_rgb = np.swapaxes(height_rgb, 0, 1)
        height_sem = np.swapaxes(height_sem, 0, 1)

        height_sem\
        = np.eye(self.num_channel)[height_sem.flatten()].reshape((height_rgb.shape[0], height_rgb.shape[1], self.num_channel))
        height_sem = torch.from_numpy(height_sem).permute(2, 0, 1).unsqueeze(0).float()

        return height_rgb, height_sem
    # ignore: start from 1        
    def append(self, quaternion, translation, observations, raw_semantics=None):
        # transform camera -> location in world
        rotation = as_rotation_matrix(quaternion)
        T_world = np.eye(4)
        T_world[0:3, 0:3] = rotation
        T_world[0:3, 3] = translation

        # get points rgbs, depths and semantics
        points = generate_pc(T_world, observations['depth'], self.fov)
        rgbs = color2local3d(observations['rgb'])
        # use system semantic observation or user semantic observation
        if raw_semantics is None:
            # get gt semantic sensor input
            semantics = observations['semantic'].flatten()
            # map from objId to category [0, num_channel-1]
            semantics = np.vectorize(lambda x: self.id2cat.get(x,\
                self.num_channel))(semantics)
            semantics -= 1
            semantics[((semantics < 0) | (semantics >= self.num_channel))] = self.num_channel - 1
        else:
            semantics = raw_semantics.flatten()
        # ignore certain category
        valid = None
        for ctg in self.ignore:
            if valid is None:
                valid = (semantics != (ctg - 1))
            else:
                valid = valid & (semantics != (ctg - 1))
        points = points[valid, :]
        rgbs = rgbs[valid, :]
        semantics = semantics[valid]
        # ignore roof/ ground points
        no_roof = (points[:, 1] < self.roof_thre) & (points[:, 1] > self.floor_thre)
        points = points[no_roof, : ]
        rgbs = rgbs[no_roof, : ]
        semantics = semantics[no_roof]

        if self.points is None:
            self.markers = [len(points)]
            self.points = points
            self.rgbs = rgbs
            self.semantics = semantics

        else:
            # remember the newest observation
            self.points = np.concatenate((self.points, points), axis=0)
            self.rgbs = np.concatenate((self.rgbs, rgbs), axis=0)
            self.semantics = np.concatenate((self.semantics, semantics), axis=0)
            self.markers.append(len(self.points))

        if self.pano:
            self.ready_counter = self.ready_counter + 1
            if self.ready_counter == 4:
                if len(self.markers) == 4:
                    self.markers = [self.markers[-1]]
                else:
                    self.markers = self.markers[:-4] + [self.markers[-1]]
                self.ready_counter = 0
        
        if not self.aggregate:
            if self.ready_counter == 0 and len(self.markers) > 1:
                self.points = self.points[self.markers[0]:]
                self.rgbs = self.rgbs[self.markers[0]:]
                self.semantics = self.semantics[self.markers[0]:]
                self.markers = self.markers[1:]

        # forget the oldest observation
        if self.ready_counter == 0 and len(self.markers) > 1 and len(self.markers) == self.memory_size + 1:
            # forget the oldest observation
            self.points = self.points[self.markers[0]:]
            self.rgbs = self.rgbs[self.markers[0]:]
            self.semantics = self.semantics[self.markers[0]:]

            # update memory markers
            for it in range(1, len(self.markers)):
                self.markers[it] = self.markers[it] - self.markers[0]
            self.markers = self.markers[1:]

class Memory_gpu(Memory):
    def __init__(self,
            aggregate,
            memory_size,
            pano,
            num_channel,
            id2cat,
            roof_thre,
            floor_thre, 
            ignore, 
            device,
            fov=np.pi / 2.,
            ):
        super(Memory_gpu, self).__init__(aggregate, memory_size,
                pano, num_channel, id2cat, roof_thre, floor_thre,
                ignore, fov)
        
        self.device = device

    def get_height_map(self, quaternion, translation, area_x, area_z, h, w):
        if self.pano:
            assert self.ready_counter ==0, "memory accumulating for pano case"
        assert len(self.markers) > 0, "Cannot view an empty memory"
        rotation = as_rotation_matrix(quaternion)
        T_world = np.eye(4)
        T_world[0:3, 0:3] = rotation
        T_world[0:3, 3] = translation
        T_world = torch.from_numpy(T_world)

        cam = torch.cat((self.points.float(), torch.ones(self.points.shape[0],
            1).to(self.device).float()),
                            dim=1).cpu()
        cam = torch.mm(cam, torch.inverse(T_world.t()).float()).to(self.device)


        pointss = cam[:, 0:3]


        round_agent = (torch.abs(pointss[:, 0]) < area_x / 2.)\
                & (torch.abs(pointss[:, 2]) < area_z / 2.)
        pointss = pointss[round_agent, :]


        scale_h = area_z / h
        scale_w = area_x / w

      
        X = (pointss[:, 0:1] + (area_x / 2.)) / float(scale_w)
        Y = pointss[:, 1:2]
        Z = (pointss[:, 2:3] + (area_z / 2.)) / float(scale_h)


        XZ_ff = torch.cat((torch.floor(X), torch.floor(Z)), dim=1)
        XZ_fc = torch.cat((torch.floor(X), torch.ceil(Z)), dim=1)
        XZ_cc = torch.cat((torch.ceil(X), torch.ceil(Z)), dim=1)
        XZ_cf = torch.cat((torch.ceil(X), torch.floor(Z)), dim=1)
       
        XZ = torch.cat((torch.cat((XZ_ff, XZ_fc), dim=0), 
            torch.cat((XZ_cc, XZ_cf), dim=0)), dim=0)
        assert h == w, "Currently only support square!"
        XZ[XZ >= h] = h - 1.
        XZ[XZ < 0.] = 0.
        Y = torch.cat((Y, Y), dim=0)
        Y = torch.cat((Y, Y), dim=0)


        XYZ = torch.cat((XZ, Y), dim=1)

        sort_ind =torch.argsort(XYZ[..., 2])

        XYZ = XYZ[sort_ind].long()
        

        height_sem = torch.ones(h, w, device=self.device) * (self.num_channel - 1)
        height_sem = height_sem.int()


        semanticss = torch.cat((self.semantics[round_agent],
            self.semantics[round_agent]), dim=0)
        semanticss = torch.cat((semanticss, semanticss), dim=0)
        semanticss = semanticss.int() 
        semanticss = semanticss[sort_ind]
        
        height_sem[XYZ[:, 0], XYZ[:, 1]] = semanticss[:]
        height_sem = torch.rot90(height_sem, 3, [0, 1])
        height_sem = torch.flip(height_sem, [1])

        height_sem = F.one_hot(height_sem.long(), num_classes=self.num_channel)


        height_sem = height_sem.reshape((h,
                w, self.num_channel))
        height_sem = height_sem.permute(2, 0, 1).unsqueeze(0).float().cpu()

        pointss, X, Y, Z, XZ_ff, XZ_fc, XZ_cc, XZ_cf =\
                pointss.cpu(), X.cpu(), Y.cpu(),\
                Z.cpu(), XZ_ff.cpu(),\
                XZ_fc.cpu(), XZ_cc.cpu(),\
                XZ_cf.cpu()

        XYZ, semanticss = XYZ.cpu(), semanticss.cpu()
        T_world, cam, round_agent = T_world.cpu(), cam.cpu(), round_agent.cpu()


        return None, height_sem
    
    def append(self, quaternion, translation, observations, raw_semantics=None):
        # transform camera -> location in world
        rotation = as_rotation_matrix(quaternion)
        T_world = np.eye(4)
        T_world[0:3, 0:3] = rotation
        T_world[0:3, 3] = translation
        T_world = torch.from_numpy(T_world)

        # get points rgbs, depths and semantics
        depth = observations['depth'][..., 0]
        depth = torch.from_numpy(depth).to(self.device)
        h, w = depth.shape
        f = float(0.5 / np.tan(self.fov / 2.) * float(w))
        x = torch.linspace(0, w-1, w)
        y = torch.linspace(0, h-1, h)
        xv, yv = torch.meshgrid(x, y)
        xv, yv = xv.t().to(self.device), yv.t().to(self.device)
        dfl = depth.reshape(-1)
        points = torch.cat((\
            (dfl * (xv.reshape(-1) - w / 2.) / f).unsqueeze(-1),\
            - (dfl * (yv.reshape(-1) - h / 2.) / f).unsqueeze(-1),\
            - dfl.unsqueeze(-1)), dim=1)

        cam = torch.cat((points, torch.ones((points.shape[0],
            1)).to(self.device)),
                    dim=1).cpu()
        cam = torch.mm(cam, T_world.t().float()).to(self.device)
        points = cam[:, 0:3]

        # use system semantic observation or user semantic observation
        if raw_semantics is None:
            # get gt semantic sensor input
            semantics = observations['semantic'].flatten()
            # map from objId to category [0, num_channel-1]
            semantics = np.vectorize(lambda x: self.id2cat.get(x,\
                self.num_channel))(semantics)
            
            semantics = torch.from_numpy(semantics).to(self.device)
            semantics -= 1
            semantics[((semantics < 0) | (semantics >= self.num_channel))] = self.num_channel - 1
        else:
            semantics\
            = torch.from_numpy(raw_semantics).to(self.device).flatten()


        # ignore certain category
        valid = None
        for ctg in self.ignore:
            if valid is None:
                valid = (semantics != (ctg - 1))
            else:
                valid = valid & (semantics != (ctg - 1))
        points = points[valid, :]

        semantics = semantics[valid]
        # ignore roof/ ground points
        no_roof = (points[:, 1] < self.roof_thre) & (points[:, 1] > self.floor_thre)
        points = points[no_roof, : ]

        semantics = semantics[no_roof]

        if self.points is None:
            self.markers = [points.shape[0]]
            self.points = points.clone()

            self.semantics = semantics.clone()

        else:
            # remember the newest observation
            self.points = torch.cat((self.points, points), dim=0)

            self.semantics = torch.cat((self.semantics, semantics), dim=0)           
            self.markers.append(self.points.shape[0])

        if self.pano:
            self.ready_counter = self.ready_counter + 1
            if self.ready_counter == 4:
                if len(self.markers) == 4:
                    self.markers = [self.markers[-1]]
                else:
                    self.markers = self.markers[:-4] + [self.markers[-1]]
                self.ready_counter = 0
        
        if not self.aggregate:
            if self.ready_counter == 0 and len(self.markers) > 1:
                self.points = self.points[self.markers[0]:]

                self.semantics = self.semantics[self.markers[0]:]
                self.markers = self.markers[1:]

        # forget the oldest observation
        if self.ready_counter == 0 and len(self.markers) > 1 and len(self.markers) == self.memory_size + 1:
            # forget the oldest observation
            self.points = self.points[self.markers[0]:]

            self.semantics = self.semantics[self.markers[0]:]

            # update memory markers
            for it in range(1, len(self.markers)):
                self.markers[it] = self.markers[it] - self.markers[0]
            self.markers = self.markers[1:]

        valid, no_roof, points, semantics = valid.cpu(), no_roof.cpu(), points.cpu(), semantics.cpu() 
        T_world, depth, dfl, xv, yv, cam = T_world.cpu(), depth.cpu(), dfl.cpu(), xv.cpu(), yv.cpu(), cam.cpu()
        
class SCNavAgent:
    def __init__(
            self, 
            device, 
            config_paths,
            flip,
            pano,
            user_semantics, 
            seg_pretrained,
            cmplt, 
            cmplt_pretrained,
            conf,
            conf_pretrained,
            distpred_pretrained,
            csr_pretrained,
            depth,
            width,
            targets, 
            aggregate, 
            memory_size, 
            num_channel,
            success_threshold, 
            collision_threshold, 
            ignore, 
            training, 
            Q_pretrained,
            state_pretrained,
            offset,
            floor_threshold, 
            lr, 
            momentum,
            weight_decay,
            gamma, 
            batch_size, 
            buffer_size,
            height,
            area_x, 
            area_z, 
            h, 
            w,
            h_new,
            w_new,
            max_step,
            navigable_base,
            success_reward,
            step_penalty,
            approach_reward,
            awayfrom_reward,
            collision_penalty,
            save_dir,
            scene_types,
            max_dist,
            double_dqn,
            TAU,
            preconf,
            seg_threshold,
            current_position,
            min_dist=0.,
            shortest=False,
            new_eval=False,
            fake_conf=False,
            discrete=False,
            att=False,
            rc=False,
            unconf=False,
            full_map=False,
            num_local=25,
            adj=11
            ):
        self.p_depth = depth
        self.p_width = width
        self.adj = adj
        self.num_local = num_local
        self.new_eval = new_eval
        assert self.new_eval, "Only support challenge setting!"
        self.shortest = shortest

        self.min_dist = min_dist
        self.TAU = TAU
        self.double_dqn = double_dqn
        self.scene_types = scene_types.split("|")

        self.max_dist = max_dist
        self.seg_threshold = seg_threshold
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.approach_reward = approach_reward
        self.awayfrom_reward = awayfrom_reward
        self.collision_penalty = collision_penalty
        
        self.rc = rc
        self.unconf = unconf
        if self.unconf:
            assert fake_conf, "currently only one-hot completion when fake confidence map is provided"
        self.batch_size = batch_size
        self.device = device
        self.save_dir = save_dir

        # create environment
        # disable habitat's metrics
        config = habitat.get_config(config_paths=config_paths)
        config.defrost()
        config.TASK.SUCCESS_DISTANCE = -float("inf")
        config.ENVIRONMENT.MAX_EPISODE_STEPS = float("inf")
        config.ENVIRONMENT.MAX_EPISODE_SECONDS = float("inf")

        config.freeze()
        self.env = habitat.Env(config=config)

        self.num_channel = num_channel
        self.ignore = ignore.split("|")
        self.ignore = [int(ig) for ig in self.ignore]
        self.offset = offset
        self.floor_threshold = floor_threshold
        self.success_threshold = success_threshold
        self.collision_threshold = collision_threshold


        self.user_semantics = user_semantics
        self.max_step = max_step
        if self.new_eval:
            self.max_step = 500
        self.navigable_base = navigable_base.split("|")
        self.navigable_base = [int(base) for base in self.navigable_base]
        if user_semantics:
            self.seg_model = ACNet(num_class = num_channel - 1)
            self.seg_model.load_state_dict(torch.load(seg_pretrained))
            self.seg_model = torch.nn.DataParallel(self.seg_model).to(device)
            self.seg_model.eval()
        self.cmplt = cmplt
        self.fake_conf = fake_conf
        self.conf = conf
        self.relation_tracking_module = RelationTrackingModule(
            csr_pretrained,
            "unweighted",
            0,
            0.5)
        if cmplt:
            self.cmplt_model = ResNet(Bottleneck, DeconvBottleneck,
                    layer_infos, num_channel).to(device)
            self.distpred_model = ResNet_dist(Bottleneck, DeconvBottleneck,
                    layer_infos, 46).to(device)

            self.cmplt_model.load_state_dict(torch.load(cmplt_pretrained))
            self.distpred_model.load_state_dict(torch.load(distpred_pretrained))

            self.cmplt_model = torch.nn.DataParallel(self.cmplt_model)
            self.distpred_model = torch.nn.DataParallel(self.distpred_model)

            self.cmplt_model.eval()
            self.distpred_model.eval()
            if conf and not self.fake_conf:
                self.conf_model = ResNet(Bottleneck, DeconvBottleneck,
                        layer_infos, num_channel, inp=1).to(device)

                self.conf_model.load_state_dict(torch.load(conf_pretrained))
                self.conf_model = torch.nn.DataParallel(self.conf_model)
                self.conf_model.eval()
        self.pano = pano
        self.discrete = discrete
        self.att = att
        if discrete:
            self.Q\
            = Q_discrete(self.num_channel, (not att and conf) or
                    self.fake_conf, preconf=preconf)
        else:
            self.Q\
            = QNet(self.num_channel, (not att and conf) or self.fake_conf, 
                    rc=rc, preconf=preconf)
            self.state_predictor \
            = State_predictor(self.num_channel, (not att and conf) or self.fake_conf,
                    rc=rc, preconf=preconf)

        if Q_pretrained != "":
            state_dict = torch.load(Q_pretrained)
            own_state = self.Q.state_dict()
            try:
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)

            except:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v

                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
        if state_pretrained != "":
            state_dict_1 = torch.load(state_pretrained)
            own_state_1 = self.state_predictor.state_dict()

            try:
                for name, param in state_dict_1.items():
                    if name not in own_state_1:
                        continue
                    own_state_1[name].copy_(param)
            except:
                new_state_dict_1 = OrderedDict()
                for k, v in state_dict_1.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict_1[name] = v

                for name, param in state_dict_1.items():
                    if name not in own_state_1:
                        continue
                    own_state[name].copy_(param)
        self.Q = torch.nn.DataParallel(self.Q).to(device)
        self.state_predictor = torch.nn.DataParallel(self.state_predictor).to(device)
        self.criterion = nn.SmoothL1Loss(reduction='sum').to(device)

        self.training = training
        if training:
            if discrete:
                self.Q_t\
                = Q_discrete(self.num_channel, (not att and conf) or self.fake_conf,
                         preconf=preconf)
            else:
                self.Q_t\
                = QNet(self.num_channel, (not att and conf) or self.fake_conf,
                        rc=rc, preconf=preconf)
            self.Q_t.load_state_dict(self.Q.module.state_dict())
            self.Q_t.eval()

            self.optimizer = optim.SGD(self.Q.parameters(), lr=lr,
                    momentum=momentum, weight_decay=weight_decay) 
            self.optimizer_s = optim.SGD(self.state_predictor.parameters(), lr=lr,
                    momentum=momentum, weight_decay=weight_decay)

            self.gamma = gamma
            self.Q_t = torch.nn.DataParallel(self.Q_t).to(device)
        else:
            self.Q.eval()

        self.targets = targets.split("|")
        self.aggregate = aggregate
        self.memory_size = memory_size

        self.replay_buffer = replay_buffer(buffer_size, save_dir,
                current_position)

        self.height = height
        self.area_x = area_x
        self.area_z = area_z
        self.h = h
        self.w = w
        self.h_new = h_new
        self.w_new = w_new
        
        self.d2x = np.zeros((480, 640, 1))
        for i in range(480):
            for j in range(640):
                self.d2x[i, j, 0] =np.sqrt(3) / 2. + (240. - i) /640.

        self.mapper = Mapper(self.device, 1024, 1024, 48., 48.,
                    self.num_channel - 1, np.pi / 2., self.ignore)
        # every episode
        self.memory = None
        self.target = None
        self.target_map = None
        self.target_objects = []
        self.target_radiuss = []
        self.target_positions = []
        self.best_path_length = float("inf")
        self.path_length = 0.
        self.eps_len = 0.
        self.reward = 0.
        self.action = None
        self.raw_semantics = None
        self.current_obs = None
        self.cmplted_obs = None
        self.conf_obs = None
        self.old_state = None
        self.state = None
        self.q_map = None

        self.obstacle = None
        self.action_list = []

        self.done = False
        self.success = False

        self.image = None
        self.depth = None
        
        self.navigable = None
        self.flip = flip
        self.episode = None
        
        self.full_map = full_map
        self.start_position = None
    
    def embedding(self, target):
        embed = torch.zeros(1, self.num_channel, self.h, self.w)
        embed[:, target, ...] = 1
        return embed

    # reset: create a new episode and restart
    # target: object name
    def reset_config(self, config):        

        
        # randomly choose a target if no specific target is given
        self.target = config['target']

        # find a valid habitat episode setting to start with
        # valid: at least one target object reachable
        # and no one target is too close


        start_position = [float(pos) for pos in config['start_position']]
        start_rotation = [float(rot) for rot in config['start_rotation']]
        self.start_position = start_position
        scene_id = config['scene_id']
        if not self.new_eval:
            start_rotation.reverse()
            self.episode = NavigationEpisode(
                goals= [],
                episode_id="0",
                scene_id=scene_id,
                start_position=start_position,
                start_rotation=start_rotation
                )
        else:
            self.episode = ObjectGoalNavEpisode(
                    goals = [],
                    episode_id='0',
                    scene_id=scene_id,
                    start_position=start_position,
                    start_rotation=start_rotation,
                    )
        self.env.episode_iterator = iter([
            self.episode
                ])
        self.env.reset()

        # pick a target candidate for this episode
        candidate_targets = [obj.category.name() for obj in
                    self.env.sim.semantic_annotations().objects if
                    obj.category.name() in self.targets]


        self.target_objects = []
        self.target_radiuss = []
        self.target_positions = []
        self.best_path_length = float("inf")

        
        for obj in self.env.sim.semantic_annotations().objects:
            if obj.category.name() == self.target:
                distance = self.env.sim.geodesic_distance(start_position,
                            obj.aabb.center)
                radius = np.sqrt(obj.aabb.sizes[0]**2\
                                + obj.aabb.sizes[2]**2)/2.


                if distance < float("inf"):
                    self.best_path_length = min(distance
                                    - self.success_threshold,
                                self.best_path_length)
                    self.target_objects.append(int(obj.id.split("_")[-1]))
                    self.target_radiuss.append(radius) 
                    self.target_positions.append(obj.aabb.center)
        
        if self.new_eval:

            self.best_path_length = config['best_path_length']


        self.env.step("LOOK_DOWN")
        roof_thre = start_position[1] + self.height/2. + self.offset
        floor_thre = start_position[1] - self.height/2. - self.floor_threshold
        
        id2cat = {int(obj.id.split("_")[-1]): obj.category.index() for obj in
                self.env.sim.semantic_annotations().objects}
        if self.full_map:

            self.mapper.reset(id2cat, roof_thre, floor_thre)
        else:

            self.memory = Memory_gpu(self.aggregate, self.memory_size, self.pano,
                self.num_channel, id2cat, roof_thre, floor_thre,
                ignore=self.ignore, device=self.device)
        self.id2cat = id2cat
        # name2id: map from "sofa" to sofa's int id
        # id: [0, 1, 2, ... 40]
        self.target = name2id[self.target]        
        # embed target as part of state
        self.target_map = self.embedding(self.target)

        self.reward = 0.
        self.action = None
        self.current_obs = None
        self.cmplted_obs = None
        self.conf_obs = None
        self.raw_semantics = None
        self.state = None
        self.old_state = None
        self.q_map = None
        self.done = False
        self.success = False

        self.image = None
        self.depth = None
        self.navigable = self.navigable_base + [self.target] 
        self.eps_len = 0.
        self.path_length = 0.
        self.relation_tracking_module.reset()
        self.view()
    # reset: create a new episode and restart
    # target: object name
    def reset(self, target=None):        

        
        # randomly choose a target if no specific target is given
        self.target = None
        if target is not None:
            self.target = target

        

        # find a valid habitat episode setting to start with
        # valid: at least one target object reachable
        # and no one target is too close


        random_heading = np.random.uniform(-np.pi, np.pi)
        start_rotation = [
                    0,
                    np.sin(random_heading / 2),
                    0,
                    np.cos(random_heading / 2),
                    ]
        while True:
            # change house by certain probability
            if self.episode is None or random.random() < self.flip:
                self.episode = random.choice(self.env.episodes)
            
                self.env.episode_iterator = iter([NavigationEpisode(
                    goals=[],
                    episode_id="0",
                    scene_id=self.episode.scene_id,
                    start_position=self.episode.start_position,
                    start_rotation=self.episode.start_rotation)]
                    )
                self.env.reset()
    
            # pick a target candidate for this episode
            candidate_targets = [obj.category.name() for obj in
                    self.env.sim.semantic_annotations().objects if
                    obj.category.name() in self.targets]
            legal_rooms = [room.aabb for room in
                    self.env.sim.semantic_annotations().regions if
                    room.category.name() in self.scene_types]
            # if no legal target to pick in this scene, remove episode
            if len(candidate_targets) == 0:
                self.episode = None
                continue
            if len(legal_rooms) == 0:
                self.episode = None
                continue
            if self.target is not None:
                if self.target not in candidate_targets:
                    self.episode = None
                    continue


            


            for trial in range(100):
                self.target_objects = []
                self.target_radiuss = []
                self.target_positions = []
                self.best_path_length = float("inf")
                edistance = float("inf")

                if target is None:
                    self.target = random.choice(candidate_targets)
                target_room = random.choice(legal_rooms)
                start_position = [target_room.center[0]
                        + (random.random() - 0.5) * abs(target_room.sizes[0]),
                        target_room.center[1] - abs(target_room.sizes[1]) / 2.,
                        target_room.center[2]
                        + (random.random() - 0.5) * abs(target_room.sizes[2])]
                if not self.env.sim.is_navigable(start_position):
                    continue
                self.start_position = start_position

                for obj in self.env.sim.semantic_annotations().objects:
                    if obj.category.name() == self.target:
                        distance = self.env.sim.geodesic_distance(start_position,
                            obj.aabb.center)
                        radius = np.sqrt(obj.aabb.sizes[0]**2\
                                + obj.aabb.sizes[2]**2)/2.
                        cedistance =self.euclidean_distance(start_position,
                                obj.aabb.center) 
                        # already in success state, illegal, start from very
                        # beginning
                        # or if min_dist is set, cannot be closer
                        if min(cedistance, distance) < radius + self.success_threshold + self.min_dist:
                            self.target_objects = []
                            self.target_radiuss = []
                            self.target_positions = []
                            self.best_path_length = float("inf")
                            edistance = float("inf")
                            break
                        if distance < float("inf"):
                            self.best_path_length = min(distance
                                    - self.success_threshold,
                                self.best_path_length)
                            self.target_objects.append(int(obj.id.split("_")[-1]))
                            self.target_radiuss.append(radius) 
                            self.target_positions.append(obj.aabb.center)
                            
                            # update shortest euclidean distance
                            edistance = min(edistance, self.euclidean_distance(start_position,
                                obj.aabb.center))
                if edistance < self.max_dist and len(self.target_objects) >= 1:
                    break
            if edistance < self.max_dist and len(self.target_objects) >= 1:
                break
        self.env.sim.set_agent_state(position=start_position,
                rotation=start_rotation)
        self.env.step("LOOK_DOWN")
        roof_thre = start_position[1] + self.height/2. + self.offset
        floor_thre = start_position[1] - self.height/2. - self.floor_threshold
        
        id2cat = {int(obj.id.split("_")[-1]): obj.category.index() for obj in
                self.env.sim.semantic_annotations().objects}
        if self.full_map:
            
            self.mapper.reset(id2cat, roof_thre, floor_thre)
        else:
            self.memory = Memory_gpu(self.aggregate, self.memory_size, self.pano,
                self.num_channel, id2cat, roof_thre, floor_thre,
                ignore=self.ignore, device=self.device)
        self.id2cat = id2cat
        # name2id: map from "sofa" to sofa's int id
        # id: [0, 1, 2, ... 40]
        self.target = name2id[self.target]        
        # embed target as part of state
        self.target_map = self.embedding(self.target)

        self.reward = 0.
        self.action = None
        self.current_obs = None
        self.cmplted_obs = None
        self.conf_obs = None
        self.raw_semantics = None
        self.state = None
        self.old_state = None
        self.q_map = None
        self.done = False
        self.success = False
        self.seen = None

        self.image = None
        self.depth = None
        self.navigable = self.navigable_base + [self.target] 
        self.eps_len = 0.
        self.path_length = 0.
        self.relation_tracking_module.reset()
        self.view()

    def map_change(self, in_semmap):
        mask = (in_semmap == in_semmap.max(dim = 1, keepdim = True)[0]).to(dtype=torch.int32)
        output = torch.mul(mask, in_semmap)
        output[output != 0] = 1
        #id = torch.argmax(in_semmap, dim=0)
        #size_1, size_2 = id.size(0), id.size(1)
        #output = torch.zeros_like(in_semmap)
        #for i in range(size_1):
        #    for j in range(size_2):
        #        output[id[i][j]][i][j] = 1
        return output

    # update agent's local Q
    def update_Q_t(self):
        self.Q_t.load_state_dict(self.Q.state_dict())

    # soft update agent's target 
    def update_Q_t_soft(self):
        old_params = {} 
        for name, params in self.Q_t.module.named_parameters():
            old_params[name] = params.clone()
        for name, params in self.Q.module.named_parameters():
            old_params[name] = old_params[name] * (1 - self.TAU)\
            + params.clone() * self.TAU
        for name, params in self.Q_t.module.named_parameters():
            params.data.copy_(old_params[name])


    def train_Q(self):

        if len(self.replay_buffer) < self.batch_size:
            return None
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask, non_final_next_states, non_final_next_relationship_bank = None, None, None
        try:
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if
                                               s is not None]).to(self.device)
            non_final_next_relationship_bank = torch.cat([s for s in batch.next_relationship_bank if
                                               s is not None]).to(self.device)
        except:
            pass
        state_batch = torch.cat(batch.state).to(self.device)
        relationship_bank_batch = torch.cat(batch.relationship_bank).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)

        state_action_values = torch.zeros(self.batch_size, device=self.device)
        q_out = self.Q(state_batch, relationship_bank_batch)
        state_action_values = q_out.view(self.batch_size, -1).gather(1,action_batch.view(self.batch_size,
                                                                    -1).long()).squeeze(-1)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask is not None and non_final_next_states is not None:
            if self.double_dqn:
                with torch.no_grad():
                    best_action \
                        = self.Q(non_final_next_states, non_final_next_relationship_bank).view(non_final_next_states.shape[0],
                                                                 -1).max(1)[1].view(non_final_next_states.shape[0],1)
                    next_state_values[non_final_mask] \
                        = self.Q_t(non_final_next_states, non_final_next_relationship_bank).view(non_final_next_states.shape[0],
                                                                   -1).gather(1, best_action).view(-1)
            else:
                next_state_values[non_final_mask] \
                    = self.Q_t(non_final_next_states, non_final_next_relationship_bank).view(non_final_next_states.shape[0],
                                                                                                    -1).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        loss_value = float(loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1.)
        self.optimizer.step()
        source = [self.h_new / 2 - 0.5, self.w_new / 2 - 0.5]
        vecs = []
        for i in range(len(action_batch)):
            a_w = action_batch[i] % self.w_new
            a_h = (action_batch[i] - a_w) / self.w_new
            d = pow(pow(a_h - source[0], 2) + pow(a_w - source[1], 2), 0.5)
            vec = torch.tensor([(a_h - source[0]) / d * 5, (a_w - source[1]) / d * 5]).unsqueeze(0)
            vecs.append(vec)
        actions = torch.cat(vecs, dim = 0)

        state_batch_ = state_batch[non_final_mask]
        relationship_bank_batch_ = relationship_bank_batch[non_final_mask]
        action_batch_ = actions[non_final_mask].unsqueeze(2).unsqueeze(3)
        action_batch_ = action_batch_.expand(non_final_mask.sum(), 2, self.h_new, self.w_new)
        next_state = self.state_predictor(state_batch_, action_batch_, relationship_bank_batch_)

        loss_s = self.criterion(non_final_next_states, next_state)
        loss_state = float(loss_s)
        self.optimizer_s.zero_grad()
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(self.state_predictor.parameters(), 1.)
        self.optimizer_s.step()
        
        return loss_value, loss_state

    # get agent's observation  
    def get_observations(self):
        return self.env.sim._sensor_suite.get_observations(self.env.sim.get_sensor_observations())
    
    def get_trinsics(self):
        quaternion = self.env._sim.get_agent_state().sensor_states['depth'].rotation
        translation = self.env._sim.get_agent_state().sensor_states['depth'].position
        return quaternion, translation
    
    def single_view_and_save(self):
        observations = self.get_observations()
        
        if self.user_semantics:
            # preprocess observation for segmentation network
            rgb = observations['rgb']
            dep = observations['depth']
            sem = observations['semantic']
            sample = {'image': rgb[..., [2, 1, 0]],
                    'depth': dep[..., 0],
                    'label': sem}
            sample = transform(sample)
            rgb = sample['image'].unsqueeze(0)
            dep = sample['depth'].unsqueeze(0)
            self.image = rgb.to(self.device)
            self.depth = dep.to(self.device)

            with torch.no_grad():
                self.raw_semantics = self.seg_model(self.image,
                        self.depth).detach().cpu()[0]
                self.raw_semantics = torch.argmax(self.raw_semantics, dim=0).numpy()
        quaternion, translation = self.get_trinsics()    
        self.memory.append(quaternion, translation, self.get_observations(), self.raw_semantics)
    
    def single_view_and_save_new(self):
    
        self.env.step("LOOK_UP")
        quaternion\
        = self.env._sim.get_agent_state().sensor_states['depth'].rotation 
        translation\
                = self.env._sim.get_agent_state().sensor_states['depth'].position
        observations\
        = self.env.sim._sensor_suite.get_observations(self.env.sim._sim.get_sensor_observations())
        if self.user_semantics:
            # preprocess observation for segmentation network
            rgb = observations['rgb']
            dep = observations['depth']
            sem = observations['semantic']
            sample = {'image': rgb[..., [2, 1, 0]],
                    'depth': dep[..., 0],
                    'label': sem}
            sample = transform(sample)
            rgb = sample['image'].unsqueeze(0)
            dep = sample['depth'].unsqueeze(0)
            self.image = rgb.to(self.device)
            self.depth = dep.to(self.device)

            with torch.no_grad():
                self.raw_semantics = self.seg_model(self.image,
                        self.depth).detach().cpu()[0]
                self.raw_semantics = torch.argmax(self.raw_semantics, dim=0).numpy()

        self.env.step("LOOK_DOWN")
        if self.user_semantics:
            self.mapper.append(quaternion, translation, observations,
                raw_semantics=self.raw_semantics)
        else:
            self.mapper.append(quaternion, translation, observations,
                raw_semantics=None)
    def view(self):
        if self.pano:
            for i in range(4):
                self.rotate(np.pi/2.)
                if self.full_map:
                    self.single_view_and_save_new()
                else:
                    self.single_view_and_save()

        else:
            if self.full_map:
                self.single_view_and_save_new()
            else:
                self.single_view_and_save()
        self.env.step("LOOK_UP")
        quaternion, translation = self.get_trinsics()        
        self.env.step("LOOK_DOWN")
        if self.full_map:
            self.current_obs = self.mapper.get_map_local_rot(quaternion, translation,
                    float(self.h),
                    float(self.w)).unsqueeze(0).float()

            
        else:
            _, self.current_obs = self.memory.get_height_map(quaternion,
                translation, self.area_x, self.area_z, self.h, self.w) 

        self.state = torch.cat((self.current_obs, self.target_map),
                dim=1)
        with torch.no_grad():
            if self.cmplt:
                self.cmplted_obs\
                    = self.cmplt_model(self.current_obs.to(self.device)).detach().cpu()
                in_semmap_2 = self.cmplted_obs[:, idxs[0], ...].unsqueeze(1)
                for l in range(1, len(idxs)):
                    in_semmap_2 = torch.cat((in_semmap_2, self.cmplted_obs[:, idxs[l], ...].unsqueeze(1)),
                                            dim=1)  # 1, 23, 128, 128
                in_semmap_2 = self.map_change(in_semmap_2).squeeze(0)  # 23, 128, 128

                self.relation_tracking_module.update_scene_representation(in_semmap_2)
                self.relationship_bank_ = self.relation_tracking_module.relationship_bank

                tmp = torch.zeros(128, 128, 512)
                if self.relationship_bank_ is not None:
                    #print(self.relationship_bank_.shape)
                    n_old = self.relationship_bank_.shape[0]  # (N, N, 512)
                    tmp[:n_old, :n_old, :] = self.relationship_bank_  # (64, 64, 512)
                self.relationship_bank_ = tmp[:64, :64, :]  # (64, 64, 512)

                if self.unconf:
                    max_idx = torch.argmax(self.cmplted_obs, 1, keepdim=True)
                    normed_cmplt\
                    = torch.FloatTensor(self.cmplted_obs.shape)
                    normed_cmplt.zero_()
                    normed_cmplt.scatter_(1, max_idx, 1)


                else:
                    normed_cmplt = F.softmax(self.cmplted_obs, dim=1)

                self.state = torch.cat((normed_cmplt, self.target_map), dim=1)
                if self.fake_conf:

                    self.conf_obs\
                            = (torch.argmax(self.current_obs,
                                    dim=1)!=self.num_channel-1).unsqueeze(1).float()

                    self.state = torch.cat((self.state, self.conf_obs), dim=1)

                elif self.conf:

                    self.conf_obs\
                            = self.conf_model(torch.cat((self.current_obs.to(self.device),
                                normed_cmplt.to(self.device)), dim=1)).detach().cpu()

                    if self.rc:
                        self.seen = (torch.argmax(self.current_obs, dim=1) !=
                                (self.num_channel - 1)).float()

                        self.seen = self.seen.unsqueeze(0)

                  #  self.conf_obs[..., torch.argmax(self.current_obs,dim=1) !=
                   #         self.num_channel - 1] = 1.


                    if not self.att:
                        self.state = torch.cat((self.state, self.conf_obs), dim=1)
                    else:

                        normed_cmplt = normed_cmplt*(1+self.conf_obs) / 2.
                        self.state = torch.cat((normed_cmplt, self.target_map),
                                dim=1)
                    if self.rc:

                        self.state = torch.cat((self.state, self.seen), dim=1) 
        # resize state for Q Net
        assert self.h == self.h_new, "no resizing for now"

        
    # shortest_path_follower.py
    # make sure this is deterministic for a (S, A) pair
    def action_picker(self, eps_threshold):
        # distribution?
        # deterministic?
        sample = random.random()
        if sample > eps_threshold:
            
            tmp = torch.argmax(self.q_map.view(-1))
            a_w = tmp % self.w_new 
            a_h = (tmp - a_w) / self.w_new
            self.action = (a_h, a_w)

        else:
            self.action = (random.randint(0, self.h_new - 1), \
                    random.randint(0, self.w_new - 1))

    def action_picker_easy(self, max_action, eps_threshold):
        # distribution?
        # deterministic?
        sample = random.random()
        if sample > eps_threshold:
            self.action = (max_action[0], max_action[1])
        else:
            self.action = (random.randint(0, self.h_new - 1), \
                           random.randint(0, self.w_new - 1))

    def reach_goal(self, c_x, c_y):
        lx = self.action[0] - int(self.adj/2.)
        ux = self.action[0] + int(self.adj/2.)
        ly = self.action[1] - int(self.adj/2.)
        uy = self.action[1] + int(self.adj/2.)
        if c_x >= lx:
            if c_x <= ux:
                if c_y >= ly:
                    if c_y <= uy:
                        return True
        return False
    def planner_path(self, eps_threshold):

        self.action_picker(eps_threshold)
        c_x = int(self.h_new / 2)
        c_y = int(self.w_new / 2)

        counter = 0
        visited = np.zeros((self.h_new,
                    self.w_new)).astype('int')
        
        a_x = max(min(self.action[0], self.h_new - 2), 1)
        a_y = max(min(self.action[1], self.w_new - 2), 1)
        visited[a_x-1:a_x + 2, a_y - 1: a_y + 2] = 1

        while not self.reach_goal(c_x, c_y):
            if counter >= self.num_local:
                break
            counter += 1
            # get up-to-date distance map at this local step
            obs = torch.argmax(self.current_obs, dim=1)
            self.obstacle = (obs == 1) | (obs == self.target) | (obs == self.num_channel - 1) 
            self.obstacle = self.obstacle[0].int().numpy()


            visited[c_x-1:c_x + 2, c_y - 1: c_y + 2] = 1
            self.obstacle[visited==1] = 1


            def add_boundary(mat):
                h, w = mat.shape
                gap = int(self.adj / 2.)
                new_mat = np.zeros((h+2 * gap,w+2 * gap))
                new_mat[gap:h+gap,gap:w+gap] = mat
                
                return new_mat
            self.obstacle = add_boundary(self.obstacle)
            tmp = np.ma.masked_values(self.obstacle * 1, 0)
            gap = int(self.adj / 2.)
            tmp[self.action[0] + gap, self.action[1] + gap] = 0
            dd = skfmm.distance(tmp, dx=1)
            dd = np.ma.filled(dd, np.max(dd) + 1)

            crop = dd[c_x:c_x + 2*gap + 1, c_y:c_y + 2*gap]
            crop[c_x + gap, c_y + gap] = np.max(dd) + 1
            (d_x, d_y) = np.unravel_index(np.argmin(crop), crop.shape)

            # rotate to face target

            angle = None
            if d_y > gap:
                if d_x < gap:
                    angle = np.arctan((d_y - gap) / (gap - d_x))

                else:
                    angle = np.arctan((d_x - gap) / (d_y - gap))
                    angle += np.pi / 2.

            elif d_x < gap:
                angle = np.arctan((gap - d_y) / (gap - d_x))
                angle = -angle

            else:

                angle = np.arctan((d_x - gap) / (gap - d_y))
         
                angle = -np.pi/2 - angle
            self.action_list.append(-angle)



            # pick a direction for the agent to step

            # execute, record reward
            # update c_x, c_y
            # update current_obs, but not using view() func
            





       
        diag = int(np.sqrt((self.action[0] - source[0])**2 + (self.action[1]
            - source[1])**2)) - 1
        if diag < 0:
            diag = 0
        for k in range(diag):
            self.action_list.append(0.)

        return self.action_list

    def planner(self, eps_threshold):

        self.action_picker(eps_threshold)
        action_list = []
        source = (self.h_new / 2 - 0.5, self.w_new / 2 - 0.5)
        
        
        # rotate to face target
        angle = None
        if self.action[1] > source[1]:
            if self.action[0] < source[0]:
                angle = np.arctan((self.action[1] - source[1]) / (source[0] - self.action[0]))

            else:
                angle = np.arctan((self.action[0] - source[0])\
                        / (self.action[1] - source[1]))
                angle += np.pi / 2.

        elif self.action[0] < source[0]:
            angle = np.arctan((source[1] - self.action[1]) / (source[0] - self.action[0]))
            angle = -angle

        else:

            angle = np.arctan((self.action[0] - source[0]) / (source[1] - self.action[1]))
     
            angle = -np.pi/2 - angle
        action_list.append(-angle)
       

        action_list.append("MOVE_FORWARD")

        return action_list

    def planner_easy(self, max_action, eps_threshold):
        self.action_picker_easy(max_action, eps_threshold)
        action_list = []
        source = (self.h_new / 2 - 0.5, self.w_new / 2 - 0.5)
        # rotate to face target
        angle = None
        if self.action[1] > source[1]:
            if self.action[0] < source[0]:
                angle = np.arctan((self.action[1] - source[1]) / (source[0] - self.action[0]))
            else:
                angle = np.arctan((self.action[0] - source[0]) \
                                  / (self.action[1] - source[1]))
                angle += np.pi / 2.
        elif self.action[0] < source[0]:
            angle = np.arctan((source[1] - self.action[1]) / (source[0] - self.action[0]))
            angle = -angle
        else:
            angle = np.arctan((self.action[0] - source[0]) / (source[1] - self.action[1]))
            angle = -np.pi / 2 - angle
        action_list.append(-angle)
        action_list.append("MOVE_FORWARD")
        return action_list

    def planner_discrete(self, randomness):
        sample = random.random()
        if sample > randomness:
            
            self.action = int(torch.argmax(self.q_map))
            

        else:
            self.action = random.randint(0, self.q_map.shape[0]-1)
        action_list = [self.action * np.pi/4., 'MOVE_FORWARD']
        return action_list

    def euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)
    
    def shortest_distance(self):
        best = float("inf")
        position = self.env.sim.get_agent_state().position
        for oid, pos in enumerate(self.target_positions):
            best = min(best, self.env.sim.geodesic_distance(position, pos)
                    - self.target_radiuss[oid] - self.success_threshold)
        return best

    def shortest_distance_to_start(self):
        position = self.env.sim.get_agent_state().position
        dis = self.env.sim.geodesic_distance(position, self.start_position)
        return dis
    
    def rotate(self, angle):
        tr = np.array([
            [np.cos(angle), 0., np.sin(angle)],
            [0., 1., 0.],
            [-np.sin(angle), 0., np.cos(angle)]
            ])

        position = self.env.sim.get_agent_state().position
        quaternion = self.env.sim.get_agent_state().rotation
        rotation = as_rotation_matrix(quaternion)

        rotation = np.dot(rotation, tr)

        self.env.sim.set_agent_state(position=position,
                rotation=from_rotation_matrix(rotation), reset_sensors=False)


    def move_path(self):
        assert False, "Pause"
        reward = 0.
    
        old_distance = self.shortest_distance()

        action = None
        distance = None



        idx = 1
        #reward += self.step_penalty
        for action in self.action_list:

            reward += self.step_penalty
            self.rotate(action)
            
            old_position = self.env.sim.get_agent_state().position
            self.env.step("MOVE_FORWARD")

            new_position = self.env.sim.get_agent_state().position

            # update real path of this episode
            distance = self.euclidean_distance(old_position,
                    new_position)
            self.path_length += distance
            self.eps_len += 1  

            self.action = (self.path[idx][0], self.path[idx][1])
            idx += 1

            # if collision happens in the course of moving
            # collision penalty
            # stop following actions
            if distance < self.collision_threshold:
                reward += self.collision_penalty
                break
            if not self.training:
                if self.eps_len >= self.max_step:
                    break

        # approaching reward
        new_distance = self.shortest_distance()
        reward += self.approach_reward * (old_distance - new_distance)
        # if not reaching terminal step num, stop itself if stop_checker True
        # judge whether this episode is successful or not


        # 4-view success measurement because the shortest path does not
        # guarantee agent's rotation after a certain step
        # as long as 1 view triggers stop checker, the agent will stop itself
        # success if a certain view triggers stop checker, and this view is
        # indeed successful 
        self.done = False
        self.success = False

        if not self.user_semantics:
            for t in range(4):
                self.rotate(np.pi/2)
                if self.success:# already successful in some view, skip checkings
                    continue
                done = self.stop_checker() # get this view's stop check result
                self.success = (done and self.success_checker()) # if this view\
                        #stops, do a success check
                self.done = (self.done or done) # if at least one view stops, stop
        else:
            success = False
            rgbs, deps = None, None  
            depths = None
            for t in range(4):
                self.rotate(np.pi/2)
                # record if at least 1 view is successful
                success = (success or self.success_checker())
                # record 4*480*640 depths
                depth = self.get_observations()['depth']
                depth = depth * self.d2x
                if depths is None:
                    depths = depth[..., 0][np.newaxis, ...]
                else:
                    depths = np.concatenate((depths, depth[..., 0][np.newaxis, ...]), axis=0)
                # store seg_model 4*... inputs
                observations = self.get_observations()

                rgb = observations['rgb']
                dep = observations['depth']
                sem = observations['semantic']
                sample = {'image': rgb[..., [2, 1, 0]],     
                        'depth': dep[..., 0],                  
                        'label': sem}
                sample = transform(sample)       
                rgb = sample['image'].unsqueeze(0)
                dep = sample['depth'].unsqueeze(0)
                rgbs = rgb if rgbs is None else torch.cat((rgbs, rgb), dim=0)
                deps = dep if deps is None else torch.cat((deps, dep), dim=0)
            # get 4*480*640 semantic output
            with torch.no_grad():
                raw_semantics = self.seg_model(rgbs.to(self.device),
                        deps.to(self.device)).detach()
                raw_semantics = torch.argmax(raw_semantics,
                        dim=1)
            depths = torch.from_numpy(depths).to(self.device)
            # get 4 * 480 * 640 check: whether each pixel is marked as successful
            check = (raw_semantics == self.target) & (depths != 0.) & (depths <= self.success_threshold)
            if torch.any(torch.sum(check.int().view(4, -1),dim=1)
                    > self.seg_threshold):
                self.done = True
            raw_semantics = raw_semantics.cpu()
            depths = depths.cpu()
            self.success = self.done and success 




        if self.success:
            reward += self.success_reward
            # if the last action caused a collision
            # but it made the episode successful
            # then do not punish it for collision
            if action is not None and distance < self.collision_threshold:
                reward -= self.collision_penalty
        # after this step of moving
        if self.eps_len >= self.max_step:
            self.done = True
        # update episode's reward record
        self.reward += reward
        

        return reward
    def move(self, action_list):
        reward = 0.
    
        old_distance = self.shortest_distance()
        old_distance_to_start = self.shortest_distance_to_start()
        self.rotate(action_list[0])
        action = None
        distance = None
        assert len(action_list) == 2, "Only support 1 step version"
        for action in action_list[1:]:
            self.eps_len += 1
            reward += self.step_penalty
            
            old_position = self.env.sim.get_agent_state().position
            self.env.step(action)
            new_position = self.env.sim.get_agent_state().position

            # update real path of this episode
            distance = self.euclidean_distance(old_position,
                    new_position)
            self.path_length += distance
            
            # if unexpected collision happens in the course of moving
            # collision penalty
            # stop following actions
            if distance < self.collision_threshold:
                reward += self.collision_penalty
                break

        # approaching reward
        new_distance = self.shortest_distance() 
        new_distance_to_start = self.shortest_distance_to_start()
        reward += self.approach_reward * (old_distance - new_distance)
        reward -= self.awayfrom_reward * (old_distance_to_start - new_distance_to_start)
        # if not reaching terminal step num, stop itself if stop_checker True
        # judge whether this episode is successful or not

        #close_enough = (new_distance <= 0.)
        close_enough = True
        
        self.done = False
        self.success = False
        if not self.user_semantics:
            for t in range(4):
                self.rotate(np.pi/2)
                if self.success:# already successful in some view, skip checkings
                    continue
                done = self.stop_checker() # get this view's stop check result
                self.success = close_enough and (done and self.success_checker()) # if this view\
                        #stops, do a success check
                self.done = (self.done or done) # if at least one view stops, stop

        if self.new_eval and (not self.user_semantics):

            if not self.user_semantics:
                for t in range(4):
                    self.rotate(np.pi/2)
                    if self.success:# already successful in some view, skip checkings
                        continue
                    done = self.stop_checker() # get this view's stop check result
                    self.success = close_enough and (done and self.success_checker()) # if this view\
                            #stops, do a success check
                    self.done = (self.done or done) # if at least one view stops, stop
            else:

                success = False
                rgbs, deps = None, None  
                depths = None
                for t in range(4):
                    self.rotate(np.pi/2)
                    # record if at least 1 view is successful
                    success = close_enough and (success or self.success_checker())
                    # record 4*480*640 depths
                    depth = self.get_observations()['depth']
                    depth = depth * self.d2x
                    if depths is None:
                        depths = depth[..., 0][np.newaxis, ...]
                    else:
                        depths = np.concatenate((depths, depth[..., 0][np.newaxis, ...]), axis=0)
                    # store seg_model 4*... inputs
                    observations = self.get_observations()

                    rgb = observations['rgb']
                    dep = observations['depth']
                    sem = observations['semantic']
                    sample = {'image': rgb[..., [2, 1, 0]],     
                            'depth': dep[..., 0],                  
                            'label': sem}
                    sample = transform(sample)       
                    rgb = sample['image'].unsqueeze(0)
                    dep = sample['depth'].unsqueeze(0)
                    rgbs = rgb if rgbs is None else torch.cat((rgbs, rgb), dim=0)
                    deps = dep if deps is None else torch.cat((deps, dep), dim=0)
                # get 4*480*640 semantic output
                with torch.no_grad():
                    raw_semantics = self.seg_model(rgbs.to(self.device),
                            deps.to(self.device)).detach()
                    raw_semantics = torch.argmax(raw_semantics,
                            dim=1)
                depths = torch.from_numpy(depths).to(self.device)
                # get 4 * 480 * 640 check: whether each pixel is marked as successful
                check = (raw_semantics == self.target) & (depths != 0.) & (depths <= self.success_threshold)
                if torch.any(torch.sum(check.int().view(4, -1),dim=1)
                        > self.seg_threshold):
                    self.done = True
                raw_semantics = raw_semantics.cpu()
                depths = depths.cpu()
                self.success = self.done and success 


        else:
            self.done = self.stop_checker()
            self.success = close_enough and (self.done and self.success_checker())


        if self.success:
            reward += self.success_reward
            # if the last action caused a collision
            # but it made the episode successful
            # then do not punish it for collision
            if action is not None and distance < self.collision_threshold:
                reward -= self.collision_penalty
        # after this step of moving
        if self.eps_len == self.max_step:
            self.done = True
        # update episode's reward record
        self.reward += reward
        
        return reward

    def visible_close_checker(self, targets, semantic, depth):

        depth = depth * self.d2x
        legal = None

        for target in targets:
            if legal is None:
                legal = (semantic != target)
            else:
                legal = legal & (semantic != target)
        legal = legal | (depth[..., 0] == 0.)
        
        check = (~legal) & (depth[..., 0] <= self.success_threshold)

        if np.any(check):
            return True
        return False
    
    def stop_checker(self):
        return self.success_checker()
    '''
    def stop_checker(self):
        if not self.user_semantics:
            return self.success_checker()
        else:
           # self.env.step("LOOK_UP")
            depth = self.get_observations()['depth']
           # self.env.step("LOOK_DOWN")
            targets = [self.target]
            
            observations = self.get_observations()
            rgb = observations['rgb']
            dep = observations['depth']
            sem = observations['semantic']
            sample = {'image': rgb[..., [2, 1, 0]],
                    'depth': dep[..., 0],                  
                    'label': sem
                    }
            sample = transform(sample)       
            rgb = sample['image'].unsqueeze(0)
            dep = sample['depth'].unsqueeze(0)
            with torch.no_grad():
                raw_semantics = self.seg_model(rgb.to(self.device),
                        dep.to(self.device)).detach()[0]
                raw_semantics = torch.argmax(raw_semantics,
                        dim=0).cpu().numpy()

            depth = depth * self.d2x
            legal = None
            for target in targets:
                if legal is None:
                    legal = (raw_semantics != target)
                else:
                    legal = legal & (raw_semantics != target)

            legal = legal | (depth[..., 0] == 0.)
            check = (~legal) & (depth[..., 0] <= self.success_threshold)

            if np.sum(check.astype('int')) > self.seg_threshold:
                return True


            return False
    '''
    def success_checker(self):

        observations = self.get_observations()

        semantics = observations['semantic']
 
        semantics = np.vectorize(lambda x: self.id2cat.get(x,\
                                self.num_channel))(semantics)

        semantics -= 1
        semantics[((semantics < 0) | (semantics >= self.num_channel))] = self.num_channel - 1
        depth = observations['depth']

        return self.visible_close_checker([self.target], semantics, depth)

    # use self.q_map, top k targets
    def action_clip(self, q_map, interval, k):
        actions = []
        a, idx1 = torch.sort(q_map.view(-1), descending=True)  # descending为False，升序，为True，降序
        a_w = idx1[0] % self.w_new
        a_h = (idx1[0] - a_w) / self.w_new
        actions.append([float(a_h), float(a_w)])
        for i in range(len(idx1)):
            a_w = idx1[i] % self.w_new
            a_h = (idx1[i] - a_w) / self.w_new
            flag = True
            for act in actions:
                dist = pow(pow((a_h - act[0]), 2) + pow((a_w - act[1]), 2), 0.5)
                if dist < interval:
                    flag = False
                    break
            if flag:
                actions.append([float(a_h), float(a_w)])
            if len(actions) == k:
                break
        return actions

    def collision_check(self, state, action):
        source = [self.h_new / 2 - 0.5, self.w_new / 2 - 0.5]
        dist = pow(pow((action[0] - source[0]), 2) + pow((action[1] - source[1]), 2), 0.5)
        direction = [action[0] - source[0], action[1] - source[1]]
        for i in range(1, int(dist)):
            delta = i / int(dist)
            x = source[0] + direction[0] * delta
            y = source[1] + direction[1] * delta
            check = state[0, :, int(x), int(y)].sum()
            if check >= 1:
                return True
        return False

    # use full_map ???
    # life_penalty, collision_penalty, distance_reward, success_reward
    def estimate_reward(self, state, action):
        est_reward = -0.01
        in_semmap = state[:, idxs[0], ...].unsqueeze(1)
        for l in range(1, len(idxs)):
            in_semmap = torch.cat((in_semmap, state[:, idxs[l], ...].unsqueeze(1)), dim = 1)
        target = classes_PONI.index(classes_SSCNav[self.target])
        embed = torch.zeros(1, 23, 128, 128).to(self.device)
        embed[:, target, ...] = 1
        in_semmap = torch.cat((self.map_change(in_semmap), embed), dim = 1)
        pred_dist = self.distpred_model(in_semmap)[0][0]

        source = [self.h_new / 2 - 0.5, self.w_new / 2 - 0.5]
        dist = pow(pow((action[0] - source[0]), 2) + pow((action[1] - source[1]), 2), 0.5) * 0.05
        action_ = [source[0] + 0.25 / dist * (action[0] - source[0]), source[1] + 0.25 / dist * (action[1] - source[1])]

        cur_dist = pred_dist[int(source[0])][int(source[1])]
        est_dist = pred_dist[int(action_[0])][int(action_[1])]

        est_reward += (cur_dist - est_dist)

        collision_check = self.collision_check(self.map_change(state[:, 0:self.num_channel - 1, ...]), action_)
        if collision_check:
            est_reward += -0.25

        if est_dist < 1:
            est_reward += 10

        return est_reward

    # self.relationship_bank_ prediction ???     # TO DO
    def V_planing(self, state, depth, width):
        current_state_value = \
            self.Q(state, self.relationship_bank_.unsqueeze(0).to(self.device)).view(state.shape[0], -1).max(1)[0].detach()

        if depth == 1:
            return current_state_value

        q_map = self.Q(state, self.relationship_bank_.unsqueeze(0).to(self.device)).detach().cpu()[0][0]

        cliped_actions = self.action_clip(q_map, 0.25 / (self.area_x / self.h), 5)
        returns = []
        for action in cliped_actions:
            source = [self.h_new / 2 - 0.5, self.w_new / 2 - 0.5]
            d = pow(pow(action[0] - source[0], 2) + pow(action[1] - source[1], 2), 0.5)
            vec = torch.tensor([(action[0] - source[0]) / d * 5, (action[1] - source[1]) / d * 5]).unsqueeze(0)
            # vec = torch.tensor([action[0] - source[0], action[1] - source[1]]).unsqueeze(0)
            action_ = vec.unsqueeze(2).unsqueeze(3)
            action_ = action_.expand(1, 2, self.h_new, self.w_new)
            next_state = self.state_predictor(state, action_, self.relationship_bank_.unsqueeze(0).to(self.device))
            next_state = (next_state == next_state.max(dim=1, keepdim=True)[0]).to(
                            dtype=torch.float32)

            current_map = state[:, 0:self.num_channel, :, :]
            reward_est = self.estimate_reward(current_map, action)     # TO DO  !!!

            next_value = self.V_planing(next_state, depth - 1, width)
            return_value = current_state_value / depth + (depth - 1) / depth * (0.99 * next_value + reward_est)
            returns.append(return_value)

        max_return = max(returns)

        return max_return
        
            
    def step(self, randomness):

        if self.eps_len == self.max_step:
            return
        if self.discrete:
            self.q_map = self.Q(self.state.to(self.device)).detach().cpu()
            assert self.q_map.shape[0] == 1, 'make sure q_map size is 1'
            self.q_map = self.q_map[0]
            self.old_state = self.state.clone()
            action_list = self.planner_discrete(randomness)
            reward = self.move(action_list)
            self.view()
            if self.training:
                self.replay_buffer.push(self.old_state,
                        torch.Tensor([self.action]), self.state if not
                    self.done else None, torch.Tensor([reward])) 
            return reward
        # think about which point to go towards 
        with torch.no_grad():
            self.q_map = self.Q(self.state.to(self.device), self.relationship_bank_.unsqueeze(0).to(self.device)).detach().cpu()[0][0]  # (128, 128)
        cliped_actions = self.action_clip(self.q_map, 0.25 / (self.area_x / self.h), self.p_width)

        self.old_state = self.state.clone()         # (1, 83, 128, 128)
        self.old_relationship_bank_  = self.relationship_bank_.clone()
        if not self.shortest:
            max_action = None
            max_value = float('-inf')
            for action in cliped_actions:
                source = [self.h_new / 2 - 0.5, self.w_new / 2 - 0.5]
                d = pow(pow(action[0] - source[0], 2) + pow(action[1] - source[1], 2), 0.5)
                vec = torch.tensor([(action[0] - source[0]) / d * 5, (action[1] - source[1]) / d * 5]).unsqueeze(0)
                #vec = torch.tensor([action[0] - source[0], action[1] - source[1]]).unsqueeze(0)
                action_ = vec.unsqueeze(2).unsqueeze(3)
                action_ = action_.expand(1, 2, self.h_new, self.w_new)
                next_state = self.state_predictor(self.state.to(self.device), action_,
                                                  self.relationship_bank_.unsqueeze(0).to(self.device))               # To Do
                next_state = (next_state == next_state.max(dim=1, keepdim=True)[0]).to(
                            dtype=torch.float32)
                #reward_est = self.estimate_reward(self.current_obs.to(self.device), action)       # To Do
                reward_est = self.estimate_reward(self.cmplted_obs.to(self.device), action)
                max_next_return = self.V_planing(next_state, self.p_depth, self.p_width)
                value = reward_est + 0.99 * max_next_return

                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError("Network is not well trained.")

            action_list = self.planner_easy(max_action, randomness)
            
            #action_list = self.planner(randomness)
            reward = self.move(action_list)

        else:
            self.planner_path(randomness)
            reward = self.move_path()

        self.view()

        if self.training:
            self.replay_buffer.push(self.old_state, self.old_relationship_bank_.unsqueeze(0),
                                    torch.Tensor([self.action[0] * self.w_new + self.action[1]]), self.state if not
                                    self.done else None, self.relationship_bank_.unsqueeze(0) if not self.done else None,
                                    torch.Tensor([reward]))

        return reward


