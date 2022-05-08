from scipy.spatial.transform import Rotation as R
import torch

def get_metric(pred, target):
    pred_angle = torch.tensor(R.from_matrix(
        pred.reshape(-1, 3, 3)
    ).as_rotvec())
    target_angle = torch.tensor(R.from_matrix(
        target.reshape(-1, 3, 3)
    ).as_rotvec())

    temp = target_angle-pred_angle
    metric = torch.norm(temp,2,-1)
    metric = torch.mean(metric)

    return metric