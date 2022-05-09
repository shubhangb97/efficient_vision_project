from scipy.spatial.transform import Rotation as R
import torch
import numpy as np

def get_metric(pred1, target1):
    pred = pred1.detach().cpu().numpy()
    target = target1.detach().cpu().numpy()
    pred_angle = torch.tensor(R.from_matrix(
        pred.reshape(-1, 3, 3)
    ).as_rotvec())
    target_angle = torch.tensor(R.from_matrix(
        target.reshape(-1, 3, 3)
    ).as_rotvec())

    temp = target_angle-pred_angle
    metric = torch.norm(temp,2,-1)
    metric = torch.mean(metric)

    return float(metric)

def get_3d_metric(pred1, target1):
    pred = pred1.permute(0,2,1,3)
    pred = pred.detach().cpu().numpy()
    target = target1.detach().cpu().numpy()
    pred = pred.reshape(pred.shape[0],pred.shape[1],pred.shape[2], 3, 3)
    target = target.reshape(target.shape[0],target.shape[1],target.shape[2], 3, 3)
    new_pred = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]-1,3))
    new_target = np.zeros((target.shape[0],target.shape[1],target.shape[2]-1,3))
    # print (pred.shape, target.shape)
    # print (pred[:,:,1,:,:].shape, pred[:,:,0,2,:].shape)
    new_pred[:,:,0] = np.matmul(pred[:,:,1,:,:],pred[:,:,0,2,:].reshape(pred.shape[0],pred.shape[1],3,1)).squeeze(-1)
    new_target[:,:,0] = np.matmul(target[:,:,1,:,:],target[:,:,0,2,:].reshape(target.shape[0],target.shape[1],3,1)).squeeze(-1)
    for i in range(2,new_pred.shape[2]+1):
        new_pred[:,:,i-1] = np.matmul(pred[:,:,i],new_pred[:,:,i-2].reshape(pred.shape[0],pred.shape[1],3,1)).squeeze(-1)
        new_target[:,:,i-1] = np.matmul(target[:,:,i],new_target[:,:,i-2].reshape(target.shape[0],target.shape[1],3,1)).squeeze(-1)
    temp = torch.tensor(new_target-new_pred)
    metric = torch.norm(temp,2,-1)
    metric = torch.mean(metric)
    return float(metric)