
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor,NearestNeighbors
import numpy as np
import torch


def fill_missing_values(x_data,x_label,y_data, k=1):

    if isinstance(x_data, torch.Tensor):
        x_data = x_data.cpu().numpy()
    if isinstance(x_label, torch.Tensor):
        x_label = x_label.cpu().numpy()
    if isinstance(y_data, torch.Tensor):
        y_data = y_data.cpu().numpy()
    

    if x_data.ndim == 3:
        n_samples, n_points, n_features = x_data.shape
        x_data = x_data.reshape(n_samples * n_points, n_features)

    if y_data.ndim == 3:
        n_samples, n_points, n_features = y_data.shape
        y_data = y_data.reshape(n_samples * n_points, n_features)
 
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_data)


    distances, indices = nn.kneighbors(y_data)
    # print(distances.shape)
    # print(indices.shape)
    avg_values = np.mean(x_label[indices], axis=1)
    # print("avg_values.shape",avg_values.shape)
    return avg_values