#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import math

from pdb import set_trace as breakpoint


class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 time_dim,
                 joints_dim
    ):
        super(ConvTemporalGraphical,self).__init__()

        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

        self.T=nn.Parameter(torch.FloatTensor(joints_dim , time_dim, time_dim))
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
        '''
        self.prelu = nn.PReLU()

        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''
    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        ## x=self.prelu(x)
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        #n  batch size c channels t time v joint w joint q time
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous()




class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True):

        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)


        self.gcn=ConvTemporalGraphical(time_dim,joints_dim) # the convolution layer

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )






        if stride != 1 or in_channels != out_channels:

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )


        else:
            self.residual=nn.Identity()


        self.prelu = nn.PReLU()



    def forward(self, x):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x=self.gcn(x)
        x=self.tcn(x)
        x=x+res
        x=self.prelu(x)
        return x




class CNN_layer(nn.Module): # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):

        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        # assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1



        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
                     ,nn.BatchNorm2d(out_channels),nn.Dropout(dropout, inplace=True)]





        self.block=nn.Sequential(*self.block)


    def forward(self, x):

        output= self.block(x)
        return output


# In[11]:


class Model(nn.Module):
    """
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider, music_dim,
                 txc_dropout,step_size, output_step_size,
                 music_as_joint,
                 num_layers,
                 bidirectional=False,
                 bias=True):

        super(Model,self).__init__()
        self.input_time_frame=input_time_frame
        self.output_time_frame=output_time_frame
        self.joints_to_consider=joints_to_consider
        self.output_step_size = output_step_size
        self.input_channels = input_channels
        self.music_as_joint = music_as_joint
        self.music_dim = music_dim
        self.st_gcnns=nn.ModuleList()
        self.txcnns=nn.ModuleList()

        self.audio_cnn = CNN_layer(music_dim,64,[3,1],txc_dropout)

        self.st_gcnns.append(ST_GCNN_layer(input_channels,64,[1,1],1,input_time_frame,
                                           joints_to_consider,st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(64,32,[1,1],1,input_time_frame,
                                               joints_to_consider+music_as_joint,st_gcnn_dropout))

        self.st_gcnns.append(ST_GCNN_layer(32,64,[1,1],1,input_time_frame,
                                               joints_to_consider+music_as_joint,st_gcnn_dropout))

        self.st_gcnns.append(ST_GCNN_layer(64,input_channels,[1,1],1,input_time_frame,
                                               joints_to_consider+music_as_joint,st_gcnn_dropout))

        self.rnn_audio_cnn = CNN_layer(music_dim,64,[step_size,1],txc_dropout)#step_size_removed

        self.D = 2 if bidirectional else 1
        self.num_layers_rnn = num_layers

                # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        self.rnn = nn.RNN(input_size=output_step_size*64,batch_first=True,
                                hidden_size=input_time_frame*input_channels*(joints_to_consider+music_as_joint),dropout=txc_dropout, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, x, x_audio, x_audio_future):
        num_gcn = 0
        for gcn in (self.st_gcnns):
            if(num_gcn == 0):
                #x_audio shape assumes is N*num_music_dim*num_time_frame*1
                x_audio = self.audio_cnn(x_audio)
            elif(num_gcn == 1):
            #x audio here is N*64*num_time_frame*1
            #x is N*64*num_time_frame*25

                x = torch.cat((x,x_audio.repeat(1,1,1,self.music_as_joint)), dim = 3)
            x = gcn(x)
            num_gcn = num_gcn+1
        #x is N*9*num_time_frame*26
        x= x.permute(0,2,1,3)
        #x is N*num_input_time_frame*9*26
        x = x.reshape((1, x.shape[0], -1))
        x = x.repeat(self.num_layers_rnn*self.D, 1, 1)

        # x_future  N*num_music_dim*num_output_time_frame*1
        #breakpoint()
        x_audio_future = self.rnn_audio_cnn(x_audio_future)
        seq_length = int(self.output_time_frame / self.output_step_size)
        # x future is now N*64*num_output_time_frame*1
        #breakpoint()
        x_audio_future = x_audio_future.permute(0,2,1,3)
        # x future is now N*num_output_time_frame*64*1
        x_audio_future = x_audio_future.reshape((x_audio_future.shape[0],seq_length,-1))
        #breakpoint()
        x,_ = self.rnn(x_audio_future, x)
        #output is to be reshaped to as this shape was input x is N*num_time_frame*9*26
        x = x.reshape((x.shape[0],seq_length,self.input_time_frame,self.input_channels,self.joints_to_consider+self.music_as_joint))
        x = x[:,:,self.input_time_frame-self.output_step_size:,:,0:-self.music_as_joint]
        #breakpoint()
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2],x.shape[3], x.shape[4] )#[:,:,:,0:-1]
        #breakpoint()
        # batch size, seq_length
        return x
