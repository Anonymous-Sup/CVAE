from fast_pytorch_kmeans import KMeans
import torch


kmeans = KMeans(n_clusters=8, mode='euclidean', verbose=1)

x = torch.randn(100000, 64, device='cuda')
labels = kmeans.fit_predict(x)