import torch
import logging
import time
import datetime
from torch import nn
from torch.utils.data import DataLoader
from trainFunc import train
from train_utils import collector,evaluateSENT,get_aspect_vector_DICT,collcate_pad_Full,ASAP_Full
from model import init_weights,TextCNNReg
from preprocess import Vocab
#Config Logger
datasetName = "train.csv"
save_name = datasetName + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
Log_Format = "%(levelname)s %(asctime)s - %(message)s"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename=save_name + '.log',
    filemode="w",
    format=Log_Format,
    level=logging.INFO)
logger = logging.getLogger()

#Environment Configuration
logger.info(torch.__version__)
import numpy as np
logger.info(np.__version__)
import sys
logger.info(sys.version)

#Hyperparameters
wordvec_length = 50
epochs = 30
random_seed = "random"
# random_seed = 500
# torch.manual_seed(random_seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# wordvec_path = "/Users/Charlie/Desktop/sgns.wiki.bigram"
gpu_boost = 4
base_lr = 1e-3
learning_rate = base_lr*gpu_boost if device != 'cpu' else base_lr
base_batch = 64
batch_size = base_batch*gpu_boost if device != 'cpu' else base_batch
pool_length = 1
num_aspects = 18
num_classes = 5
min_freq = 3
# wiki_vec_dict = 352217
# input_path = "/kaggle/input/meituan-asap/"
input_path = ""
loss_func = nn.L1Loss()
kernel_sizes, num_channels = [3, 4, 5], [100, 100, 100]
# embed_size, num_hiddens, num_layers, devices = 50, 5, 2,'cpu'
#Load Data
vocab = Vocab(reserved_tokens=['<unk>'],min_freq=min_freq,dict_path="word_freq.txt")
aspect_indexes,aspect_dict = get_aspect_vector_DICT(vocab)
aspect_indexes = torch.tensor(aspect_indexes).to(device)
training_data = ASAP_Full(input_path + datasetName,vocab)
test_data = ASAP_Full(input_path + "dev.csv",vocab);
train_dataloader = DataLoader(training_data, batch_size=batch_size, \
collate_fn=collcate_pad_Full, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, collate_fn=collcate_pad_Full)
#Prepare Model
# NN = TextCNN(len(vocab),wordvec_length,kernel_sizes,num_channels)
NN = TextCNNReg(len(vocab),wordvec_length,kernel_sizes,num_channels)
NN.apply(init_weights)
optimizer = torch.optim.Adam(NN.parameters(),lr=learning_rate)
#Start Training
print('=' * 7 + "Training Begin" + '=' * 7)
print(
    f"total epoch: {epochs}\ndevice: {device}\nbatch size: {batch_size}\nloss func:{loss_func}\nlr: {learning_rate}\nrandom_seed: {random_seed}\ndict size: {len(vocab)}\nwordvec length: {wordvec_length}\nfilter num: {num_channels}\nfilter size: {kernel_sizes}\npool length: {pool_length}")
logger.info(
    f"total epoch: {epochs}\ndevice: {device}\nbatch size: {batch_size}\nloss func:{loss_func}\nlr: {learning_rate}\nrandom_seed: {random_seed}\ndict size: {len(vocab)}\nwordvec length: {wordvec_length}\nfilter num: {num_channels}\nfilter size: {kernel_sizes}\npool length: {pool_length}")
logger.info(NN)
train(NN,train_dataloader,test_dataloader,loss_func,optimizer,epochs,device,save_name,logger)