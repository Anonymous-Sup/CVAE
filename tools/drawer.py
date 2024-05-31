
import torch
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class tSNE_plot():
    def __init__(self, num_query, feat_norm=True, trainplot=False):
        super(tSNE_plot, self).__init__()
        self.num_query = num_query
        self.trainplot = trainplot
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.styles = []
        self.U = []

    def update(self, output):  # called once for each batch
        feat, pid, styles = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid.cpu()))
        # styles is str like 'rgb' or 'sketch' 
        self.styles.extend(np.asarray(styles))
    
    def update_U(self, U):
        self.U.append(U.cpu())


    def compute(self, run):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        
        feats = feats.detach().numpy()
        # Ensure pids is a numpy array
        pids = np.asarray(self.pids)

        # Create a mask for query and gallery samples
        if not self.trainplot:
            mask = np.concatenate([np.zeros(self.num_query), np.ones(len(self.pids) - self.num_query)])
            print("The number of query sketch samples:", self.num_query)
            print("The number of gallery rgb samples:", len(self.pids) - self.num_query)
        else:
            # make mask according to the style, 'sketch' set 0 and 'rgb' set 1, else raise error
            mask = np.zeros(len(self.styles))
            mask = np.where(np.array(self.styles) == 'rgb', 1, mask)
            # mask = np.asarray([0 if style == 'sketch' else 1 for style in self.styles])
            # print the number of sketch and rgb
            print('The number of sketch samples:', np.sum(mask == 0))
            print('The number of rgb samples:', np.sum(mask == 1))

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(feats)
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in np.unique(mask):
            ax.scatter(tsne_results[mask == i, 0], tsne_results[mask == i, 1], label='sketch' if i == 0 else 'rgb', alpha=0.6, s=50)
        ax.legend()
        ax.grid(True)
        ax.set_title('t-SNE of Features')

        # Save the plot
        if self.trainplot:
            run['train/tsne/{}'.format("0-q_g_train")].append(fig)
        else:
            run['test/tsne/{}'.format("0-q_g_test")].append(fig)
        # del 
        plt.close(fig)


        # Plot t-SNE according to PID
        fig, ax = plt.subplots(figsize=(6, 6))
        # Filter to only the first 10 PIDs
        unique_pids = np.unique(pids)[:20]
        filtered_mask = np.isin(pids, unique_pids)

        # Filter the features and PIDs
        filtered_feats = feats[filtered_mask]
        filtered_pids = pids[filtered_mask]

        # Compute t-SNE on the filtered features
        tsne = TSNE(n_components=2, random_state=42, perplexity=19)
        tsne_results = tsne.fit_transform(filtered_feats)
        
        for idx, pid in enumerate(unique_pids):
            pid_mask = filtered_pids == pid
            ax.scatter(tsne_results[pid_mask, 0], tsne_results[pid_mask, 1], label='id-{}'.format(pid), alpha=0.6, s=50)
        ax.legend()
        ax.grid(True)
        ax.set_title('t-SNE of Features by PID')

        # Save the plot
        if self.trainplot:
            run['train/tsne/{}'.format("0-pid_train")].append(fig)
        else:
            run['test/tsne/{}'.format("0-pid_test")].append(fig)
        # del
        plt.close(fig)



        # drawing U 
        U = torch.cat(self.U, dim=0)
        U = U.detach().numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in np.unique(mask):
            ax.scatter(U[mask == i, 0], U[mask == i, 1], label='sketch' if i == 0 else 'rgb', alpha=0.6, s=50)
        ax.legend()
        ax.grid(True)
        ax.set_title('t-SNE of U')

        # Save the plot
        if self.trainplot:
            run['train/tsne/{}'.format("0-U_train")].append(fig)
        else:
            run['test/tsne/{}'.format("0-U_test")].append(fig)
        # del
        plt.close(fig)
