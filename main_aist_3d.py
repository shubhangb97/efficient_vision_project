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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)


model = Model(
	args.input_dim,args.input_n,
	args.output_n,args.st_gcnn_dropout,
	args.joints_to_consider,
	args.n_tcnn_layers,args.tcnn_kernel_size,
	args.tcnn_dropout).to(device)

print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

model_name='aist_final_3d_o'+str(args.output_n)+'_i'+str(args.input_n)+'_cnn'+str(args.n_tcnn_layers)+'frames_ckpt'

def train():


	optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-05)

	if args.use_scheduler:
			scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

	train_loss = []
	val_loss = []
	dataset = datasets.Datasets(args.p3d_pth, args.audio_pth,args.input_n,args.output_n,args.skip_rate)
	print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
	data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

	vald_dataset = datasets.Datasets(args.p3d_val_pth, args.audio_val_pth,args.input_n,args.output_n,args.skip_rate)
	print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
	vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

	dim_used = np.arange(225) # 25 * 9
	# assumed the x y z are 9*1 with 0 pads for each dimension
	running_metric = 0
	divi = 0
	curr_train_loss = 0
	curr_val_loss = 0
	for epoch in range(args.n_epochs):
		running_loss=0
		n=0
		model.train()
		for batch in enumerate(data_loader):
			cnt, batch = batch
			batch, _ = batch
			# print(len(batch))
			batch=batch.to(device)
			batch_dim=batch.shape[0]
			n+=batch_dim
			assert batch.shape[2] == 225
			# print(batch.shape)

			# sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
			sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//args.input_dim,args.input_dim).permute(0,3,1,2)
			sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//args.input_dim,args.input_dim)

			# print(sequences_train.shape, sequences_gt.shape)


			optimizer.zero_grad()
			# change
			sequences_predict=model(sequences_train).permute(0,1,3,2)
			#print(sequences_predict.shape, sequences_gt.shape)
			loss=mpjpe_error(sequences_predict,sequences_gt)
			metric = get_3d_metric(sequences_predict,sequences_gt)
			# print(metric)
			running_metric += metric * sequences_predict.shape[0]
			divi += sequences_predict.shape[0]

			if cnt % 200 == 0:
				print('[%d, %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))

			loss.backward()
			if args.clip_grad is not None:
				torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

			optimizer.step()
			running_loss += loss*batch_dim
		print ("BATCHES DONE: ", running_metric/divi)
		curr_train_loss = running_loss.detach().cpu()/n
		train_loss.append(curr_train_loss)
		model.eval()
		with torch.no_grad():
			running_loss=0
			n=0
			for batch in enumerate(vald_loader):
				cnt, batch = batch
				batch,_ = batch
				batch=batch.to(device)
				batch_dim=batch.shape[0]
				n+=batch_dim


				# sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
				# sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//3,3)
				sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//args.input_dim,args.input_dim).permute(0,3,1,2)
				sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//args.input_dim,args.input_dim)

				sequences_predict=model(sequences_train).permute(0,1,3,2)

				assert (sequences_predict.shape == sequences_gt.shape)
				loss=mpjpe_error(sequences_predict,sequences_gt)
				if cnt % 200 == 0:
									print('[%d, %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))
				running_loss+=loss*batch_dim
			curr_val_loss = running_loss.detach().cpu()/n
			val_loss.append(curr_val_loss)
		if args.use_scheduler:
			scheduler.step()


		if (epoch+1)%5==0:
			print('----saving model-----')
			torch.save(model.state_dict(),os.path.join(args.model_path,model_name+"e%d_v%.3f_t%.3f" % (epoch,curr_val_loss, curr_train_loss)))


			plt.figure(1)
			plt.plot(train_loss, 'r', label='Train loss')
			plt.plot(val_loss, 'g', label='Val loss')
			plt.legend()
			plt.show()

def test():

	model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
	model.eval()
	accum_loss=0
	n_batches=0 # number of batches for all the sequences
	actions=define_actions(args.actions_to_consider)

	dim_used =np.arange(219)
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
				batch, _ = batch
				batch=batch.to(device)
				batch_dim=batch.shape[0]
				n+=batch_dim


				all_joints_seq=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]

				# sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
				# sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :]
				sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//args.input_dim,args.input_dim).permute(0,3,1,2)
				sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,len(dim_used)//args.input_dim,args.input_dim)


				sequences_predict=model(sequences_train).permute(0,1,3,2).contiguous().view(-1,args.output_n,len(dim_used))



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
