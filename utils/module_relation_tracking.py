import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from models import moco2_module_old, moco2_module
from utils.utils_csr import (check_none_or_empty, get_device, load_lightning_inference, render_adj_matrix)
from torchvision.transforms.transforms import ToTensor
import numpy as np
from sklearn.cluster import MeanShift
import itertools
import cv2
import matplotlib.pyplot as plt

class RelationTrackingModule(object):
    def __init__(
            self,
            relation_tracking_model_path,
            averaging_strategy,
            device_relation_tracking,
            cos_sim_match_threshold) -> None:

        super().__init__()

        self.relation_tracking_model = None
        self.device = get_device(device_relation_tracking)

        if not check_none_or_empty(relation_tracking_model_path):
            self.relation_tracking_model = load_lightning_inference(
                relation_tracking_model_path, moco2_module.MocoV2).encoder_q.to(self.device)
        else:
            raise ValueError(
                'relation_tracking_model_path should never be None or empty')

        self.averaging_strategy = averaging_strategy

        self.cos_sim_match_threshold = cos_sim_match_threshold
        self.reset()

    def reset(self):
        self.update_count = 0
        self.feature_bank = None
        self.feature_match_counts = None
        self.relationship_bank = None
        self.relationship_match_counts = {}

    def update_scene_representation(self, state_):

        x, edge_pairings, num_self = self.create_batch(state_)
        edge_features = {}
        x_instance = torch.cat((x['local_map'], x['mask_1'], x['mask_2']), 1).to(self.device)
        if x_instance.shape[0] > 0:
            feat_instance = None
            i = 0
            while i < x_instance.shape[0]:
                if feat_instance is None:
                    feat_instance = self.relation_tracking_model(x_instance[i:i+100])
                else:
                    feat_instance = torch.cat((feat_instance, self.relation_tracking_model(x_instance[i:i+100])), 0)
                i += 100

            feat_instance = F.normalize(feat_instance, dim=1).cpu()
            query_features = feat_instance[:num_self]

            for i, pairing in enumerate(edge_pairings):
                edge_features[pairing] = feat_instance[i + num_self]

            if self.feature_bank is None:
                self.initialize_scene_representation(query_features, edge_features)
            else:
                self.match_scene_representation_pred(query_features, edge_features)

            assert self.relationship_bank.shape[0] == self.relationship_bank.shape[1]
            assert self.relationship_bank.shape[2] == self.feature_bank.shape[0]
            assert self.relationship_bank.shape[0] == self.feature_bank.shape[1]

            # update the relationship with the main diagonal self features
            for i in range(self.feature_bank.shape[1]):
                self.relationship_bank[i, i] = self.feature_bank[:, i]

    def cluster(self, local_map):
        masks = {}
        cnt = 0
        for ch in range(2, local_map.size(0) - 1):
            channel = local_map[ch, ...].numpy().astype(np.uint8)
            vis = cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)
            img = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
            #plt.subplot(1, 2, 1)
            #plt.imshow(img)

            #kernel = np.ones((2, 2), np.uint8)
            #erode_thresh = cv2.erode(img, kernel)
            #plt.subplot(1, 2, 2)
            #plt.imshow(erode_thresh)
            #plt.show()

            retval, labels, stats, centriids = cv2.connectedComponentsWithStats(img, connectivity = 8)
            label = np.unique(labels)
            if len(label) == 0:
                continue
            for i in range(1, len(label)):
                mask = (labels == i).astype(np.uint8)
                nonzero = np.count_nonzero(mask)
                if nonzero < 20:
                    continue
                masks[cnt] = mask
                cnt += 1
        return masks
    '''
            index = np.argwhere(channel > 0)
            ms = MeanShift(bandwidth=20)
            if len(index) == 0:
                continue
            ms.fit(index)
            labels = ms.labels_
            num = len(list(set(labels)))
            for i in range(num):
                lab = np.argwhere(labels == i).squeeze(1)
                idx = index[lab, :]
                h, l = idx[:, 0], idx[:, 1]
                mask = np.zeros_like(channel)
                mask[h, l] = 1
                masks[cnt] = mask
                cnt += 1
                # plt.subplot(1, 3, 1)
                # plt.imshow(mask)
                # plt.show()
        return masks
        '''

    #state_: (23, 128, 128)
    def create_batch(self, state_):     #(N, 23, 128, 128) (N, 1, 128, 128) (N, 1, 128, 128)

        masks = self.cluster(state_)
        edge_pairings = list(itertools.permutations(masks.keys(), 2))
        self_pairings = [(i, i) for i in masks]
        num_self = len(self_pairings)
        keys = self_pairings + edge_pairings

        mask_1 = torch.zeros((len(keys), 1, state_.size(1), state_.size(2)))
        mask_2 = torch.zeros((len(keys), 1, state_.size(1), state_.size(2)))
        local_map = torch.zeros((len(keys), state_.size(0), state_.size(1), state_.size(2)))
        for i, k in enumerate(keys):
            mask_1[i] = torch.from_numpy(masks[k[0]]).unsqueeze(0)
            mask_2[i] = torch.from_numpy(masks[k[1]]).unsqueeze(0)
            local_map[i] = torch.clone(state_)

        return {'mask_1': mask_1, 'mask_2': mask_2, 'local_map': local_map}, edge_pairings, num_self

    def initialize_scene_representation(self, query_features, edge_features):
        self.update_count += 1

        self.feature_bank = torch.transpose(query_features, 0, 1)
        self.relationship_bank = torch.zeros( query_features.shape[0], query_features.shape[0], query_features.shape[1])
        for pair in edge_features:
            self.relationship_bank[pair[0], pair[1]] = edge_features[pair]

        self.feature_match_counts = torch.ones(
            self.feature_bank.shape[1])

    def match_scene_representation_pred(self, query_features, edge_features):
        self.update_count += 1

        # start with all features being unmatched with the history
        unmatched_queries = set([i for i in range(query_features.shape[0])])

        # create a reward matrix between current observation and the feature bank
        sim = torch.matmul(query_features, self.feature_bank)

        # hungarian matching to get the maximal assignment
        query_idx, history_idx = linear_sum_assignment(sim.numpy(), maximize=True)

        assert len(query_idx) == len(history_idx)

        det_idx_to_cluster_idx = {i: None for i in range(query_features.shape[0])}

        for i in range(len(query_idx)):
            cluster_number = history_idx[i]
            if sim[query_idx[i], history_idx[i]] > self.cos_sim_match_threshold:
                # considered a match if the sim is greater than the threshold
                det_idx_to_cluster_idx[query_idx[i]] = cluster_number

                # remove from the unmatched queries set
                unmatched_queries.remove(query_idx[i])

                if self.averaging_strategy == 'weighted':
                    # weighted average to integrate the query feature into the history
                    self.weighted_average_self_feature(
                        cluster_number, query_features, query_idx[i])
                else:
                    # unweighted average, which has the affect of weighting newer observations more
                    self.unweighted_average_self_feature(
                        cluster_number, query_features, query_idx[i])

                # re-normalize
                self.feature_bank[:, cluster_number] = F.normalize(
                    self.feature_bank[:, cluster_number], dim=0)

        # get the queries that have not matched
        unmatched_queries = list(unmatched_queries)

        l = self.feature_bank.size(1)
        for u in unmatched_queries:
            det_idx_to_cluster_idx[u] = l
            l = l + 1

        # expand the relationship bank as necessary
        num_new_clusters = len(unmatched_queries)
        if num_new_clusters != 0:
            n_old = self.relationship_bank.shape[0]
            n_new = n_old + num_new_clusters
            tmp = torch.zeros(n_new, n_new, query_features.shape[1])
            tmp[:n_old, :n_old, :] = self.relationship_bank
            self.relationship_bank = tmp

        # append features to the feature bank
        new_features = torch.transpose(query_features[unmatched_queries], 0, 1)
        self.feature_bank = torch.cat((self.feature_bank, new_features), 1)

        self.feature_match_counts = torch.cat((self.feature_match_counts, torch.ones(len(unmatched_queries))), 0)

        for pair in edge_features:
            # fill in the edge feature representations
            ith, jth = det_idx_to_cluster_idx[pair[0]], det_idx_to_cluster_idx[pair[1]]
            assert ith != jth
            if (ith, jth) not in self.relationship_match_counts:
                # norm should be 1, so if this is the case we have a new relation and need to just fill it with the edge feature
                self.relationship_bank[ith, jth] = edge_features[pair]
                self.relationship_match_counts[(ith, jth)] = 1
            elif self.averaging_strategy == 'weighted':
                raise NotImplementedError('gotta write this still')
            else:
                self.relationship_match_counts[(ith, jth)] += 1
                self.relationship_bank[ith, jth] = (
                    self.relationship_bank[ith, jth] + edge_features[pair]) / 2
                self.relationship_bank[ith, jth] = F.normalize(
                    self.relationship_bank[ith, jth], dim=0)


    def weighted_average_self_feature(self, cluster_number, query_features, instance_number):
        # weighted average to integrate the query feature into the history
        self.feature_bank[:, cluster_number] = self.feature_bank[:, cluster_number] * \
            self.feature_match_counts[cluster_number] + query_features[instance_number]
        self.feature_match_counts[cluster_number] += 1
        self.feature_bank[:, cluster_number] /= self.feature_match_counts[cluster_number]

    def unweighted_average_self_feature(self, cluster_number, query_features, instance_number):
        self.feature_bank[:, cluster_number] = self.feature_bank[:, cluster_number] + query_features[instance_number]
        self.feature_match_counts[cluster_number] += 1
        self.feature_bank[:, cluster_number] /= 2
