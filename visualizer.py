import torch
import numpy as np
from smplx import SMPL
from tfrecord.torch.dataset import TFRecordDataset
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
from scipy.spatial.transform import Rotation as R

def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden



def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest
    

def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl

tfrecord_path = "/Users/eash/UIUC/CS 598- Vision/project/tf_sstables/aist_generation_train_v2_tfrecord-00002-of-00020"
dataset = TFRecordDataset(tfrecord_path, None, None)
loader = torch.utils.data.DataLoader(dataset, batch_size=1)
data = next(iter(loader))

vid_name = data['motion_name'][0].numpy()
vid_name = ''.join([chr(x) for x in vid_name])
print("Vidoe name:", vid_name)

a=data['motion_sequence'].reshape((data['motion_sequence_shape'][0][0].item(), data['motion_sequence_shape'][0][1].item()))
b=a.numpy()
c=np.concatenate((np.zeros((b.shape[0], 6)), b), axis=1)
motion=c.reshape(1, 444, 225)

smpl_model = SMPL(model_path="/Users/eash/Downloads/SMPL_python_v.1.1.0/smpl/models/SMPL_MALE.pkl", gender='MALE', batch_size=1)

smpl_poses, smpl_trans = recover_to_axis_angles(motion)
smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
keypoints3d = smpl_model.forward(
    global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
    body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
    transl=torch.from_numpy(smpl_trans).float(),
).joints.detach().numpy()   # (seq_len, 24, 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

data=keypoints3d[0]
graph = ax.scatter(data[:, 0], data[:, 2], data[:, 1])

def update_graph(num):
    data=keypoints3d[num]
    graph._offsets3d = (data[:, 0], data[:, 2], data[:, 1])
    title.set_text('3D Test {}, time={}'.format(vid_name, num))

ani = matplotlib.animation.FuncAnimation(fig, update_graph, keypoints3d.shape[0], 
                               interval=33, blit=False)

plt.show()
