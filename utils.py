import torch
import matplotlib.pyplot as plt
import math
import numpy as np

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


def plot_histogram(run, tensor, title):
    
    # 创建一个图像对象
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 在这个图像对象上绘制直方图
    ax.hist(tensor.detach().cpu().numpy().flatten(), bins=30, alpha=0.6, color='g')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True)

    # 保存图像到一个缓冲区
    # plt.savefig("temp_plot.png", format='png')
    run["train/histograms/{}".format(title)].append(fig)
    # 关闭plt，避免重复显示图像
    plt.close(fig)

# plot for each second dim in the tensor
def plot_pair(run, tensor, title):
        
    # 创建一个图像对象
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 在这个图像对象上绘制直方图
    for i in range(tensor.size(1)):
        ax.hist(tensor[:,i].detach().cpu().numpy().flatten(), bins=30, alpha=0.6, label="dim_{}".format(i))
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    ax.legend()

    # 保存图像到一个缓冲区
    # plt.savefig("temp_plot.png", format='png')
    run["train/histograms/{}".format(title)].append(fig)
    # 关闭plt，避免重复显示图像
    plt.close(fig)

def plot_pair_seperate(run, tensor, title):
    # Calculate the number of rows and columns for the subplots
    num_dims = tensor.size(1)
    num_cols = 4
    num_rows = math.ceil(num_dims / num_cols)

    # Create a new figure and subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows))

    # Flatten the axes array to make iterating over it easier
    axs = axs.flatten()

    # Iterate over each dimension in the second axis of the tensor
    for i in range(num_dims):
        # Plot the histogram for the current dimension on the corresponding subplot
        axs[i].hist(tensor[:,i].detach().cpu().numpy().flatten(), bins=30, alpha=0.6, label="dim_{}".format(i))
        axs[i].set_title("{} - dim_{}".format(title, i))
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        axs[i].grid(True)
        axs[i].legend()

    # Remove any unused subplots
    for i in range(num_dims, num_rows * num_cols):
        fig.delaxes(axs[i])

    # Show the plot
    plt.tight_layout()
    # plt.show()

    run["train/histograms/{}".format(title)].append(fig)
    # Save the figure to a file
    # fig.savefig("plot.png", format='png')

    # Close the figure to free up memory
    plt.close(fig)


def plot_correlation_matrix(run, tensor, title):
    # Convert the tensor to a numpy array
    array = tensor.detach().cpu().numpy()
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(array, rowvar=False)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 6))
    # Create a heatmap from the correlation matrix
    cax = ax.matshow(corr_matrix, cmap='coolwarm')

    # Add a colorbar to the figure
    fig.colorbar(cax)
    # Set the title of the plot
    ax.set_title(title)

    # Show the plot
    # plt.show()
    # Save the figure to a file
    # fig.savefig("correlation_matrix.png", format='png')
    run["train/histograms/{}".format(title)].append(fig)

    # Close the figure to free up memory
    plt.close(fig)

def plot_scatter_2D(run, tensor, title):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 6))
    # Create a scatter plot from the tensor
    ax.scatter(tensor[:, 0].detach().cpu().numpy(), tensor[:, 1].detach().cpu().numpy(), alpha=0.6)
    # Set the title of the plot
    ax.set_title(title)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Domain index')
    ax.grid(True)

    # Show the plot
    # plt.show()
    # Save the figure to a file
    # fig.savefig("scatter_plot.png", format='png')
    run["train/histograms/{}".format(title)].append(fig)

    # Close the figure to free up memory
    plt.close(fig)

# plot_scatter函数用于绘制散点图, for tensor(64)
def plot_scatter_1D(run, tensor, title):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 4))
    # Plot the scatter plot
    ax.scatter(range(tensor.size(0)), tensor.detach().cpu().numpy(), alpha=0.6)

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    # Show the plot
    # plt.show()
    run["train/histograms/{}".format(title)].append(fig)
    # Save the figure to a file
    # fig.savefig("scatter_plot.png", format='png')

    # Close the figure to free up memory
    plt.close(fig)

class EarlyStopping:
    def __init__(self, patience=5, threshold=0.01):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, kl_loss):
        if kl_loss < self.best_loss:
            self.best_loss = kl_loss
            self.counter = 0
        elif kl_loss >= self.best_loss + self.threshold:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False