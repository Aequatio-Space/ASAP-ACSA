import torch
from torch import nn
from torch.utils.data import DataLoader
from train_utils import get_aspect_vector_DICT,collcate_pad,collcate_pad_Full\
,ASAP_Full,ASAP_ASPECT_DICT,evaluateSENT_CLS,CrossEntropyLoss_LS,evaluateTSV\
,evaluateASPECTForFull
from model import GCAE_ACSA_RL,GCAE_ACSA_RL_CLS,TextCNN,GCAE_ACSA
from preprocess import Vocab,read_words
#Config Logger
datasetName = "train.csv"
#Hyperparameters
wordvec_length = 50
min_freq = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_classes = 3
pool_length = 50
num_aspects = 18
input_path = ""
loss_func = CrossEntropyLoss_LS(label_smooth=0.10,class_num=num_classes)
kernel_sizes, num_channels = [3, 4, 5], [64 ,64 ,64]
#Load Data
#This version can only load full dict
# tokens = read_words(input_path + "train_SENT.tsv")
# vocab = Vocab(tokens,reserved_tokens=['<unk>'],min_freq=min_freq)
vocab = Vocab(reserved_tokens=['<unk>'],min_freq=min_freq,dict_path = "word_freq.txt")
aspect_indexes,aspect_dict = get_aspect_vector_DICT(vocab)
aspect_indexes = torch.tensor(aspect_indexes).to(device)
# test_data = ASAP_ASPECT_DICT("dev.tsv",vocab,aspect_dict,"val")
test_data = ASAP_Full(input_path + "test.csv",vocab);
# test_dataloader = DataLoader(test_data, batch_size=1, collate_fn=collcate_pad)
test_dataloader = DataLoader(test_data, batch_size=1, collate_fn=collcate_pad_Full)
NN = GCAE_ACSA(len(vocab),wordvec_length,kernel_sizes,num_channels,aspect_indexes,num_classes)
state_dict_path = "/Users/Charlie/Downloads/trian.tsv2022-06-04 00_55_50.pt"
# evaluateTSV(NN, state_dict_path, test_dataloader, loss_func, device, outputPath="ACSA_result.txt",join=False)
evaluateASPECTForFull(NN,state_dict_path,test_dataloader, loss_func, aspect_indexes,device,outputPath="ACSA-test-result.tsv")
