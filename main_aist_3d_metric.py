import os
from utils import aist_dataset as datasets
from torch.utils.data import DataLoader
from model import *
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
from metric import get_metric
from metric import get_3d_metric
from metric_fid_np import get_fid_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)


model = Model(
	args.input_dim,args.input_n,
	args.output_n,args.st_gcnn_dropout,
	args.joints_to_consider,
	args.n_tcnn_layers,args.tcnn_kernel_size,
	args.tcnn_dropout).to(device)

print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# model_name='aist_final_3d_o'+str(args.output_n)+'_i'+str(args.input_n)+'_cnn'+str(args.n_tcnn_layers)+'frames_ckpt'
model_name = 'aist_final_3d_o10_i10_cnn4frames_ckpte19_v0.193_t0.224'

def train():


	model.load_state_dict(torch.load(os.path.join(args.model_path,model_name),map_location=torch.device('cpu')))
	model.eval()

	dim_used =np.arange(225)

	running_fidk=[]
	running_fidg=[]

	vald_dataset = datasets.Datasets(args.p3d_val_pth, args.audio_val_pth,args.input_n,args.output_n,args.skip_rate)
	print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
	vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

	for batch in enumerate(vald_loader):
		with torch.no_grad():

			cnt, batch = batch
			batch, _ = batch
			batch=batch.to(device)

			# sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
			# sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :]
			sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//args.input_dim,args.input_dim).permute(0,3,1,2)
			sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//args.input_dim,args.input_dim)


			sequences_predict=model(sequences_train).permute(0,1,3,2).contiguous().view(-1,args.output_n,len(dim_used))

			FID_k,FID_g = get_fid_metric(sequences_predict,sequences_gt)
			if FID_k < 100000 and FID_g < 100000:
				running_fidk.append(FID_k)
				running_fidg.append(FID_g)
				# print (FID_k, FID_g)
	
	print (np.mean(running_fidk),np.mean(running_fidg))

def test():
	pass
	# model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
	# model.eval()

	# dim_used =np.arange(225)

	# running_fidk=[]
	# running_fidg=[]

	# vald_dataset = datasets.Datasets(args.p3d_val_pth, args.audio_val_pth,args.input_n,args.output_n,args.skip_rate)
	# print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
	# vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

	# for batch in enumerate(vald_loader):
	# 	with torch.no_grad():

	# 		cnt, batch = batch
	# 		batch, _ = batch
	# 		batch=batch.to(device)

	# 		# sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
	# 		# sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :]
	# 		sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//args.input_dim,args.input_dim).permute(0,3,1,2)
	# 		sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//args.input_dim,args.input_dim)


	# 		sequences_predict=model(sequences_train).permute(0,1,3,2).contiguous().view(-1,args.output_n,len(dim_used))

	# 		FID_k,FID_g = get_fid_metric(sequences_predict,sequences_gt)
	# 		if FID_k < 100000 and FID_g < 100000:
	# 			running_fidk.append(FID_k)
	# 			running_fidg.append(FID_g)
	# 			print (FID_k, FID_g)
	
	# print (np.mean(running_fidk),np.mean(running_fidg))


if __name__ == '__main__':

	if args.mode == 'train':
		train()
	elif args.mode == 'test':
		test()
	elif args.mode=='viz':
		model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
		model.eval()
		visualize(args.input_n,args.output_n,args.visualize_from,args.data_dir,model,device,args.n_viz,args.skip_rate,args.actions_to_consider)
