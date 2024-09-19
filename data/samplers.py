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
        # self.length = 0
        # for pid in self.pids:
        #     idxs = self.index_dic[pid]
        #     num = len(idxs)
        #     if num < self.num_instances:
        #         num = self.num_instances
        #     self.length += num - num % self.num_instances
        
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            # Ensure the calculation of length considers full batches
            full_batches = num // self.num_instances
            self.length += full_batches * self.num_instances
            # Add one more batch if there are remaining samples
            if num % self.num_instances != 0:
                self.length += self.num_instances

    def __iter__(self):
        list_container = []

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            
            if len(idxs) < self.num_instances:
    #             assert 1==0, "This should not happen"
    #             if isinstance(self.index2cluster[0], str):
    #                 assert self.num_instances > 3, "num_instances must be greater than 2 for string cluster_id"
    # #                Handle case when cluster_id is a string
    #                 rgb_idxs = [idx for idx in idxs if self.index2cluster[idx] == 'rgb']
    #                 sketch_idxs = [idx for idx in idxs if self.index2cluster[idx] == 'sketch']
                    
    #                 # Ensure there is at least one 'rgb' and one 'sketch'
    #                 if len(rgb_idxs) == 0 or len(sketch_idxs) == 0:
    #                     raise ValueError(f"Identity {pid} does not have both 'rgb' and 'sketch' instances.")

    #                 # Sample 1 from rgb and 2 from sketch
    #                 idxs = random.sample(rgb_idxs, 1) + random.sample(sketch_idxs, 2)

    #                 remaining_num = self.num_instances - 3
    #                 remaining_idxs = np.random.choice(idxs, size=remaining_num, replace=True)
    #                 idxs.extend(remaining_idxs)
    #             else:
    #                 idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    # print(batch_idxs)
                    list_container.append(batch_idxs)
                    batch_idxs = []
            
            if len(batch_idxs) > 0 and len(batch_idxs) < self.num_instances:
                while len(batch_idxs) < self.num_instances:
                    batch_idxs.append(random.choice(idxs))
                list_container.append(batch_idxs)

        random.shuffle(list_container)

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        return self.length
    
   
class FewshotSampler(Sampler):
    """
    Randomly sample N classes, and for each class, 
    randomly sample K instances. This process is repeated for `num_episodes` times.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_classes (int): number of classes per episode.
    - num_instances (int): number of instances per class in an episode.
    - num_episodes (int): number of episodes to sample.
    """
    def __init__(self, data_source, num_classes, num_instances, num_episodes):
        
        self.data_source = data_source
        self.num_classes = num_classes  # N classes
        self.num_instances = num_instances  # K instances per class
        self.num_episodes = num_episodes  # Number of episodes

        self.index_dic = defaultdict(list)
        # Create a dictionary mapping each class (pid) to its image indices
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())  # List of unique class IDs (pids)

    def __iter__(self):
        for _ in range(self.num_episodes):  # Iterate over episodes
            episode_idxs = []  # Collect indices for this episode
            
            # Randomly select `num_classes` (N) from available pids (with replacement allowed)
            selected_pids = random.sample(self.pids, self.num_classes)
            
            for pid in selected_pids:
                idxs = self.index_dic[pid]
                
                # Randomly sample `num_instances` (K) from this class
                if len(idxs) >= self.num_instances:
                    selected_idxs = np.random.choice(idxs, size=self.num_instances, replace=False)
                else:
                    # If not enough samples, sample with replacement
                    selected_idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                
                random.shuffle(selected_idxs) 
                episode_idxs.extend(selected_idxs)
            
            # Instead of yielding all indices at once, yield them one by one
            for idx in episode_idxs:
                yield idx

    def __len__(self):
        # estimate number of examples in an epoch
        return self.num_episodes*self.num_classes*self.num_instances
