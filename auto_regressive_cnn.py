import os
from utils import aist_dataset as datasets
from torch.utils.data import DataLoader
from aist_model import *
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
from smplx import SMPL
import torch
import numpy as np
from utils.loss_funcs import *
from utils.data_utils import define_actions
from utils.h36_3d_viz import visualize
from utils.parser import args
from pdb import set_trace as breakpoint
from metric import get_metric, get_3d_metric
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.animation

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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)




model = Model_cnn(
	args.input_dim,args.input_n,
	args.output_n,args.st_gcnn_dropout,
	args.joints_to_consider, args.music_dim,
	args.tcnn_dropout,args.step_size, args.output_step_size,
	args.music_as_joint, args.n_tcnn_layers, args.tcnn_kernel_size).to(device)

# model = Model(
# 	args.input_dim,args.input_n,
# 	args.output_n,args.st_gcnn_dropout,
# 	args.joints_to_consider, args.music_dim,
# 	args.tcnn_dropout,args.step_size, args.output_step_size,
# 	args.music_as_joint, args.n_tcnn_layers).to(device)

print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

prefix = "train"#"val"
p3_path = "../data/p3d_"+prefix+".pth"
audio_path = "../data/audio_"+prefix+".pth"
model_path = "./checkpoints/CKPT_3D_AIST/aist_final_cnn_music_o10_i10_m12_cnn1_s5_os2frames_ckpte24_v0.828_t0.217_m0.000"
start_frame = 20

p3_data = torch.load(p3_path)
audio_data = torch.load(audio_path)

vid_name =  'gLO_sBM_c01_d15_mLO5_ch10'
# vid_name =  'gJB_sBM_c01_d09_mJB4_ch10'
print("Vidoe name:", vid_name)

# a=data['motion_sequence'].reshape((data['motion_sequence_shape'][0][0].item(), data['motion_sequence_shape'][0][1].item()))
a = p3_data[vid_name]
fs = np.arange(start_frame, a.shape[0], 2)
p3d_, audio_ = torch.cat((torch.zeros(fs.shape[0], 6), a[fs]), axis=1), audio_data[vid_name][fs]


p3_batch = p3d_.to(device).unsqueeze(0)
music_batch = audio_.to(device).unsqueeze(0)

dim_used = np.arange(225) # 25 * 9
music_dim_used = np.arange(35)

model.load_state_dict(torch.load(os.path.join(model_path), map_location=device))
model.eval()

p3_train = p3_batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//args.input_dim,args.input_dim)

for i in range(args.auto_regressive_steps):
    sequences_train = p3_train[:, -args.input_n:, :].permute(0,3,1,2)
    music_train = music_batch[:, p3_train.shape[1]-args.input_n:p3_train.shape[1], music_dim_used].permute(0,2,1).unsqueeze(3)
    music_future=music_batch[:, p3_train.shape[1]:p3_train.shape[1]+args.output_n, music_dim_used].permute(0,2,1).unsqueeze(3)

    sequences_predict=model(sequences_train, music_train, music_future).permute(0,1,3,2)

    # sequences_predict = sequences_predict.permute(0, 2, 3, 1)
    p3_train = torch.cat((p3_train, sequences_predict), axis=1)

c = p3_train.detach().cpu().numpy()

motion = c.reshape((c.shape[0],c.shape[1],-1))

smpl_model = SMPL(model_path="../data/SMPL_MALE.pkl", gender='MALE', batch_size=1)

smpl_poses, smpl_trans = recover_to_axis_angles(motion)
smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
keypoints3d = smpl_model.forward(
    global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
    body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
    transl=torch.from_numpy(smpl_trans).float(),
).vertices.detach().numpy()   # (seq_len, 24, 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
ax.grid(False)
plt.axis('off')

data=keypoints3d[0]
graph = ax.scatter(data[:, 0], data[:, 2], data[:, 1])

def update_graph(num):
    data=keypoints3d[num]
    graph._offsets3d = (data[:, 0], data[:, 2], data[:, 1])
    if num < args.input_n:
        graph.set(color = 'r')
    elif num < args.input_n+args.output_n:
        graph.set(color = 'b')
    else:
        graph.set(color = 'y')
    title.set_text('Auto-regressive with CNN {}, time={}'.format(vid_name, num))

ani = matplotlib.animation.FuncAnimation(fig, update_graph, keypoints3d.shape[0],
                               interval=60, blit=False)

plt.show()


f = r"../data/animation_autocnn_o"+str(args.output_n)+"_i"+str(args.input_n)+"_"+vid_name+".gif"
writergif = matplotlib.animation.PillowWriter(fps=30) 
# writervideo = matplotlib.animation.FFMpegWriter(fps=60) 
ani.save(f, writer=writergif)