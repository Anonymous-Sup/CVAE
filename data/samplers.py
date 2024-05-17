import copy
import random
import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        self.index2cluster = {}
        for index, (_, pid, _, cluster_id) in enumerate(data_source):
            self.index_dic[pid].append(index)
            self.index2cluster[index] = cluster_id # 'rgb' or 'sketch
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        list_container = []

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            
            if isinstance(self.index2cluster[0], str):
                assert self.num_instances > 2, "num_instances must be greater than 2 for string cluster_id"
#                Handle case when cluster_id is a string
                rgb_idxs = [idx for idx in idxs if self.index2cluster[idx] == 'rgb']
                sketch_idxs = [idx for idx in idxs if self.index2cluster[idx] == 'sketch']
                
                # Ensure there is at least one 'rgb' and one 'sketch'
                if len(rgb_idxs) == 0 or len(sketch_idxs) == 0:
                    raise ValueError(f"Identity {pid} does not have both 'rgb' and 'sketch' instances.")

                # Sample at least one from each cluster
                sampled_idxs = random.sample(rgb_idxs, 1) + random.sample(sketch_idxs, 1)

                remaining_num = self.num_instances - 2
                remaining_idxs = [idx for idx in idxs if idx not in sampled_idxs]

                if len(remaining_idxs) < remaining_num:
                    remaining_idxs = np.random.choice(remaining_idxs, size=remaining_num, replace=True)
                else:
                    remaining_idxs = random.sample(remaining_idxs, remaining_num)

                sampled_idxs.extend(remaining_idxs)
                random.shuffle(sampled_idxs)

            else:
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                sampled_idxs = idxs[:self.num_instances]

            list_container.append(sampled_idxs)

            # batch_idxs = []
            # for idx in idxs:
            #     batch_idxs.append(idx)
            #     if len(batch_idxs) == self.num_instances:
            #         list_container.append(batch_idxs)
            #         batch_idxs = []

        random.shuffle(list_container)

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        return self.length