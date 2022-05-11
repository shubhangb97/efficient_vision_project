# import vedo
import torch
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import linalg

# import glob
# import tqdm
from smplx import SMPL

# See https://github.com/google/aistplusplus_api/ for installation 
from kinetic import extract_kinetic_features
from manual import extract_manual_features


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return torch.tensor(iden.astype(np.float32))


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = torch.linalg.svd(rotmats)
    r_closest = torch.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = torch.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = torch.sign(det)
    r_closest = torch.matmul(torch.matmul(u, iden), vh)
    return r_closest
    

def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        torch.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    ).detach().numpy()
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return torch.tensor(axis_angles), transl


def visualize(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()   # (seq_len, 24, 3)

    bbox_center = (
        keypoints3d.reshape(-1, 3).max(axis=0)
        + keypoints3d.reshape(-1, 3).min(axis=0)
    ) / 2.0
    bbox_size = (
        keypoints3d.reshape(-1, 3).max(axis=0) 
        - keypoints3d.reshape(-1, 3).min(axis=0)
    )
    world = vedo.Box(bbox_center, bbox_size[0], bbox_size[1], bbox_size[2]).wireframe()
    vedo.show(world, axes=True, viewup="y", interactive=0)
    for kpts in keypoints3d:
        pts = vedo.Points(kpts).c("red")
        plotter = vedo.show(world, pts)
        if plotter.escaped: break  # if ESC
        time.sleep(0.01)
    vedo.interactive().close()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    Code apapted from https://github.com/mseitzer/pytorch-fid

    Copyright 2018 Institute of Bioinformatics, JKU Linz
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    mu and sigma are calculated through:
    ```
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    ```
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = torch.atleast_1d(mu1)
    mu2 = torch.atleast_1d(mu2)

    sigma1 = torch.atleast_2d(sigma1)
    sigma2 = torch.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not torch.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if torch.iscomplexobj(covmean):
        if not torch.allclose(torch.diagonal(covmean).imag, 0, atol=1e-3):
            m = torch.max(torch.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = torch.trace(covmean)

    return (diff.dot(diff) + torch.trace(sigma1)
            + torch.trace(sigma2) - 2 * tr_covmean)


def extract_feature(motion, smpl_model, mode="kinetic"):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    # print (smpl_poses.shape, smpl_trans.shape)
    # print(smpl_poses.shape, smpl_trans.shape)
    smpl_poses = torch.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = torch.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    # smpl_poses = smpl_poses.reshape(-1,24,3)
    # smpl_trans = smpl_trans.reshape(-1,3)
    # smpl_poses = smpl_poses[0,:,:,:]
    # smpl_trans = smpl_trans[0,:,:,:]
    # keypoints3d = smpl_model.forward(
    #     global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
    #     body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
    #     transl=torch.from_numpy(smpl_trans).float(),
    # ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)
    smpl_model = smpl_model.float()
    keypoints3d = smpl_model.forward(
        global_orient=smpl_poses[:, 0:1].float(),
        body_pose=smpl_poses[:, 1:].float(),
        transl=smpl_trans.float(),
    ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)

    if mode == "kinetic":
      feature = extract_kinetic_features(keypoints3d)
    elif mode == "manual":
      feature = extract_manual_features(keypoints3d)
    else:
      raise ValueError("%s is not support!" % mode)
    return feature # (f_dim,)


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = torch.stack(feature_list1)
    feature_list2 = torch.stack(feature_list2)

    # normalize the scale
    mean = torch.mean(feature_list1, axis=0)
    std = torch.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=torch.mean(feature_list1, axis=0), 
        sigma1=torch.cov(feature_list1, rowvar=False),
        mu2=torch.mean(feature_list2, axis=0), 
        sigma2=torch.cov(feature_list2, rowvar=False),
    )
    return dist


# if __name__ == "__main__":
#     import glob
#     import tqdm
#     from smplx import SMPL

#     # get cached motion features for the real data
#     real_features = {
#         "kinetic": [np.load(f) for f in glob.glob("./data/aist_features/*_kinetic.npy")],
#         "manual": [np.load(f) for f in glob.glob("./data/aist_features/*_manual.npy")],
#     }

#     # set smpl
#     smpl = SMPL(model_path="/mnt/data/smpl/", gender='MALE', batch_size=1)

#     # get motion features for the results
#     result_features = {"kinetic": [], "manual": []}
#     result_files = glob.glob("outputs/*.npy")
#     for result_file in tqdm.tqdm(result_files):
#         result_motion = np.load(result_file)[None, ...]  # [1, 120 + 1200, 225]
#         # visualize(result_motion, smpl)
#         result_features["kinetic"].append(
#             extract_feature(result_motion[:, 120:], smpl, "kinetic"))
#         result_features["manual"].append(
#             extract_feature(result_motion[:, 120:], smpl, "manual"))
    
#     # FID metrics
#     FID_k = calculate_frechet_feature_distance(
#         real_features["kinetic"], result_features["kinetic"])
#     FID_g = calculate_frechet_feature_distance(
#         real_features["manual"], result_features["manual"])
    
#     # Evaluation: FID_k: ~38, FID_g: ~27
#     # The AIChoreo paper used a bugged version of manual feature extractor from 
#     # fairmotion (see here: https://github.com/facebookresearch/fairmotion/issues/50)
#     # So the FID_g here does not match with the paper. But this value should be correct.
#     # In this aistplusplus_api repo the feature extractor bug has been fixed.
#     # (see here: https://github.com/google/aistplusplus_api/blob/main/aist_plusplus/features/manual.py#L50)
#     print('\nEvaluation: FID_k: {:.4f}, FID_g: {:.4f}\n'.format(FID_k, FID_g))


def get_fid_metric(pred1,target1):
    print ("NEW")
    smpl = SMPL(model_path="../data/SMPL_MALE.pkl", gender='MALE', batch_size=1)
    pred1 = pred1.permute(0,2,1,3)
    # smpl = None
    pred1 = pred1.reshape(pred1.shape[0],pred1.shape[1],-1)
    target1 = target1.reshape(target1.shape[0],target1.shape[1],-1)
    # print (pred1.shape, target1.shape)
    # gt_kin = extract_feature(target1, smpl, "kinetic")
    # gt_man = extract_feature(target1, smpl, "manual")
    # res_kin = extract_feature(pred1, smpl, "kinetic")
    # res_man = extract_feature(pred1, smpl, "manual")
    gt_kin = []
    gt_man = []
    res_kin = []
    res_man = []
    for i in range(pred1.shape[0]):
        gt_kin.append(torch.tensor(extract_feature(target1[i,:,:].reshape(1,target1.shape[1],225), smpl, "kinetic")))
        gt_man.append(torch.tensor(extract_feature(target1[i,:,:].reshape(1,target1.shape[1],225), smpl, "manual")))
        res_kin.append(torch.tensor(extract_feature(pred1[i,:,:].reshape(1,pred1.shape[1],225), smpl, "kinetic")))
        res_man.append(torch.tensor(extract_feature(pred1[i,:,:].reshape(1,pred1.shape[1],225), smpl, "manual")))

    FID_k = calculate_frechet_feature_distance(gt_kin, res_kin)
    FID_g = calculate_frechet_feature_distance(gt_man, res_man)

    return FID_k, FID_g