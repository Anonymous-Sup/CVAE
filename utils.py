import torch
import matplotlib.pyplot as plt

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