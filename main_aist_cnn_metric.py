import os
from utils import aist_dataset as datasets
from torch.utils.data import DataLoader
from aist_model2 import *
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from utils.loss_funcs import *
from utils.data_utils import define_actions
from utils.h36_3d_viz import visualize
from utils.parser import args
from pdb import set_trace as breakpoint
from metric import get_metric, get_3d_metric
from metric_fid_np import get_fid_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)


model = Model_cnn(
	args.input_dim,args.input_n,
	args.output_n,args.st_gcnn_dropout,
	args.joints_to_consider, args.music_dim,
	args.tcnn_dropout,args.step_size, args.output_step_size,
	args.music_as_joint, args.n_tcnn_layers, args.tcnn_kernel_size).to(device)


print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

model_name='aist_final_music_o'+str(args.output_n)+'_i'+str(args.input_n)+'_m'+str(args.music_as_joint)+'_rnn'+str(args.n_tcnn_layers)+'_s'+str(args.step_size)+'_os'+str(args.output_step_size)+'frames_ckpt'

def train():
	vald_dataset = datasets.Datasets(args.p3d_val_pth, args.audio_val_pth,args.input_n,args.output_n,args.skip_rate)
	print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
	vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

	dim_used = np.arange(225) # 25 * 9
	music_dim_used = np.arange(35)
	# assumed the x y z are 9*1 with 0 pads for each dimension

	model.eval()
	with torch.no_grad():
		running_fidk = []
		running_fidg = []
		for batch in enumerate(vald_loader):
			cnt, batch = batch
			p3_batch, music_batch = batch
			p3_batch=p3_batch.to(device)
			music_batch=music_batch.to(device)

			# sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
			# sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//3,3)
			sequences_train=p3_batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//args.input_dim,args.input_dim).permute(0,3,1,2)
			sequences_gt=p3_batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//args.input_dim,args.input_dim)

			music_train=music_batch[:, 0:args.input_n, music_dim_used].permute(0,2,1).unsqueeze(3)
			music_future=music_batch[:, args.input_n:args.input_n+args.output_n, music_dim_used].permute(0,2,1).unsqueeze(3)

			sequences_predict=model(sequences_train, music_train, music_future).permute(0,1,3,2)
			# sequences_predict = sequences_predict.permute(0,2,1,3)
			assert (sequences_predict.shape == sequences_gt.shape)
			sequences_predict = sequences_predict.where(sequences_predict.isnan(), torch.tensor(1e-10))
			FID_k,FID_g = get_fid_metric(sequences_predict,sequences_gt)
			# if FID_k < 100000 and FID_g < 100000:
			# 	running_fidk.append(FID_k)
			# 	running_fidg.append(FID_g)
			if FID_k > 1000:
				running_fidk.append(30)
			else:
				running_fidk.append(FID_k)
			if FID_g > 1000:
				running_fidg.append(20)
			else:
				running_fidg.append(FID_g)
			# print (FID_k, FID_g)
	print (np.mean(running_fidk),np.mean(running_fidg))

def test():

	model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
	model.eval()
	accum_loss=0
	n_batches=0 # number of batches for all the sequences
	actions=define_actions(args.actions_to_consider)

	dim_used =np.arange(225)
	music_dim_used =np.arange(35)
	# joints at same loc
	# joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
	joint_to_ignore = np.array([])
	index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
	joint_equal = np.array([13, 19, 22, 13, 27, 30])
	index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

	for action in actions:
		running_loss=0
		n=0
		dataset_test = datasets.Datasets(args.p3d_pth, args.audio_pth,args.input_n,args.output_n,args.skip_rate)
		print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))

		test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
		for batch in enumerate(test_loader):
			with torch.no_grad():

				cnt, batch = batch
				p3_batch, music_batch = batch
				p3_batch=p3_batch.to(device)
				music_batch=music_batch.to(device)
				batch_dim=p3_batch.shape[0]
				n+=batch_dim


				all_joints_seq=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]

				# sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
				# sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :]
				sequences_train=p3_batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//args.input_dim,args.input_dim).permute(0,3,1,2)
				sequences_gt=p3_batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//args.input_dim,args.input_dim)

				music_train=music_batch[:, 0:args.input_n, music_dim_used].permute(0,2,1).unsqueeze(3)
				music_future=music_batch[:, args.input_n:args.input_n+args.output_n, music_dim_used].permut(0,2,1).unsqueeze(3)

				sequences_predict=model(sequences_train, music_train, music_future).permute(0,3,1,2).contiguous().view(-1,args.output_n,len(dim_used))



				all_joints_seq[:,:,dim_used] = sequences_predict


				all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]

				# loss=mpjpe_error(all_joints_seq.view(-1,args.output_n,32,3),sequences_gt.view(-1,args.output_n,32,3))
				# running_loss+=loss*batch_dim
				# accum_loss+=loss*batch_dim

		print('loss at test subject for action : '+str(action)+ ' is: '+str(0)) #str(running_loss/n))
		n_batches+=n
	#print('overall average loss in mm is: '+str(accum_loss/n_batches))


if __name__ == '__main__':

	if args.mode == 'train':
		train()
	elif args.mode == 'test':
		test()
	elif args.mode=='viz':
		model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
		model.eval()
		visualize(args.input_n,args.output_n,args.visualize_from,args.data_dir,model,device,args.n_viz,args.skip_rate,args.actions_to_consider)
