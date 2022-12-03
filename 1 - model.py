from torchtext.vocab import Vectors
import torch
import torch.nn.functional as F
from torch import nn
class GCAEReg(nn.Module):
    def __init__(self, vec_file, max_seq_length, dict_size, wordvec_length,aspect_indexes, filter_num, window_size, pool_length, device):
        super(GCAEReg, self).__init__()
        self.pool_length = pool_length
        if vec_file is not None:
            self.wordvec = Vectors(name=vec_file)
            self.embedding = nn.Embedding(300000,300,device = device).from_pretrained(self.wordvec.vectors,freeze=False)
        else:
            self.embedding = nn.Embedding(dict_size,wordvec_length,device = device)
        self.aspectFC = nn.Linear(wordvec_length,filter_num)
        #aspectFC should output filter_num!
        self.wordConv1 = nn.Sequential(nn.Conv1d(wordvec_length,filter_num,window_size,padding=2),
            nn.Dropout(0.3),
        )
        self.wordConv2 = nn.Sequential(nn.Conv1d(wordvec_length,filter_num,window_size,padding=2),
            nn.Dropout(0.3),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(self.pool_length)
        self.classifyFC = nn.Sequential(nn.Flatten(1,2),
                                        nn.Linear(filter_num*self.pool_length,1),)
        self.aspectIndexes = aspect_indexes
    def forward(self, x):
        input = self.embedding(x).transpose(1,2)
        aspect_feat = self.wordConv1(input)
        senti_feat = self.wordConv2(input)
        aspect_feat = F.relu(aspect_feat)
        out = torch.tanh(senti_feat) * aspect_feat
        out = self.classifyFC(self.maxpool(out))
        return out
class GCAEReg_MultiConv(nn.Module):
    def __init__(self, vec_file, dict_size, wordvec_length,aspect_indexes, num_channels, kernel_sizes, paddings, pool_length, device):
        super(GCAEReg_MultiConv, self).__init__()
        self.pool_length = pool_length
        if vec_file is not None:
            self.wordvec = Vectors(name=vec_file)
            self.embedding = nn.Embedding(300000,300,device = device).from_pretrained(self.wordvec.vectors,freeze=False)
        else:
            self.embedding = nn.Embedding(dict_size,wordvec_length,device = device)
        #aspectFC should output filter_num!
        self.wordConv1 = nn.ModuleList()
        self.wordConv2 = nn.ModuleList()
        for c, k, p in zip(num_channels, kernel_sizes, paddings):
            self.wordConv1.append(nn.Conv1d(wordvec_length, c, k, padding=p))
            self.wordConv2.append(nn.Conv1d(wordvec_length, c, k, padding=p))
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(self.pool_length)
        self.classifyFC = nn.Sequential(nn.Flatten(1,2),
                                        nn.Linear(sum(num_channels)*self.pool_length,1),)
        self.aspectIndexes = aspect_indexes
    def forward(self, x):
        input = self.embedding(x).transpose(1,2)
        aspect_feat = torch.cat([self.relu(conv(input)) for conv in self.wordConv1],dim=1)
        senti_feat = torch.cat([torch.tanh(conv(input)) for conv in self.wordConv2],dim=1)
        out = senti_feat * aspect_feat
        out = self.classifyFC(self.maxpool(out))
        return out
class GCAE_ACSA_RL(nn.Module):
    def __init__(self, vec_file, dict_size, wordvec_length,aspect_indexes, num_channels, kernel_sizes, paddings, pool_length, num_aspects, device):
        super(GCAE_ACSA_RL, self).__init__()
        self.pool_length = pool_length
        self.aspect_indexes = aspect_indexes
        self.num_aspects = num_aspects
        if vec_file is not None:
            self.wordvec = Vectors(name=vec_file)
            self.embedding = nn.Embedding(300000,300,device = device).from_pretrained(self.wordvec.vectors,freeze=False)
        else:
            self.embedding = nn.Embedding(dict_size,wordvec_length,device = device)
        #aspectFC should output filter_num!
        self.wordConv1 = nn.ModuleList()
        self.wordConv2 = nn.ModuleList()
        # self.aspectFC = nn.ModuleList()
        for c, k, p in zip(num_channels, kernel_sizes, paddings):
            self.wordConv1.append(nn.Conv1d(wordvec_length, c, k, padding=p))
            self.wordConv2.append(nn.Conv1d(wordvec_length, c, k, padding=p))
            # self.aspectFC.append(nn.Linear(wordvec_length,c))
        self.relu = nn.ReLU()
        self.aspectFC = nn.Linear(wordvec_length,num_channels[0])
        self.maxpool = nn.AdaptiveMaxPool1d(self.pool_length)
        self.classifyFC = nn.Sequential(nn.Flatten(1,2),
                                        nn.Linear(sum(num_channels)*self.pool_length,1))
        self.aspectIndexes = aspect_indexes
    def forward(self, x, aspect_label=None):
        input = self.embedding(x).transpose(1,2)
        aspect_vectors = self.embedding(self.aspect_indexes)
        if aspect_label is not None:
            senti_feat = torch.cat([torch.tanh(conv(input))*torch.tanh((aspect_label.to(torch.float) @ self.aspectFC(aspect_vectors))).unsqueeze(2) for conv in self.wordConv2], dim=1)
        else:
            senti_feat = torch.cat([torch.tanh(conv(input)) for conv in self.wordConv2], dim=1)
        aspect_feat = torch.cat([self.relu(conv(input)) for conv in self.wordConv1],dim=1)
        out = senti_feat * aspect_feat
        out = self.classifyFC(self.maxpool(out))
        return out
class GCAE_ACSA_RL_CLS(nn.Module):
    def __init__(self, vec_file, dict_size, wordvec_length,aspect_indexes, num_channels, kernel_sizes, paddings, pool_length, num_aspects, num_classes, device):
        super(GCAE_ACSA_RL_CLS, self).__init__()
        self.pool_length = pool_length
        self.aspect_indexes = aspect_indexes
        self.num_aspects = num_aspects
        if vec_file is not None:
            self.wordvec = Vectors(name=vec_file)
            self.embedding = nn.Embedding(300000,300,device = device).from_pretrained(self.wordvec.vectors,freeze=False)
        else:
            self.embedding = nn.Embedding(dict_size,wordvec_length,device = device)
        #aspectFC should output filter_num!
        self.wordConv1 = nn.ModuleList()
        self.wordConv2 = nn.ModuleList()
        # self.aspectFC = nn.ModuleList()
        for c, k, p in zip(num_channels, kernel_sizes, paddings):
            self.wordConv1.append(nn.Conv1d(wordvec_length, c, k, padding=p))
            self.wordConv2.append(nn.Conv1d(wordvec_length, c, k, padding=p))
            # self.aspectFC.append(nn.Linear(wordvec_length,c))
        self.relu = nn.ReLU()
        self.aspectFC = nn.Sequential(nn.Linear(wordvec_length,num_channels[0]),
                                      )
        self.maxpool = nn.AdaptiveMaxPool1d(self.pool_length)
        self.classifyFC = nn.Sequential(nn.Flatten(1,2),
                                        nn.Linear(sum(num_channels)*self.pool_length,num_classes),
                                        )
        self.aspectIndexes = aspect_indexes
    def forward(self, x, aspect_label=None):
        input = self.embedding(x).transpose(1,2)
        aspect_vectors = self.embedding(self.aspect_indexes)
        if aspect_label is not None:
            senti_feat = torch.cat([torch.tanh(conv(input))*torch.tanh((aspect_label.to(torch.float) @ self.aspectFC(aspect_vectors))).unsqueeze(2) for conv in self.wordConv2], dim=1)
        else:
            senti_feat = torch.cat([torch.tanh(conv(input)) for conv in self.wordConv2], dim=1)
        aspect_feat = torch.cat([self.relu(conv(input)) for conv in self.wordConv1],dim=1)
        out = senti_feat * aspect_feat
        out = self.classifyFC(self.maxpool(out))
        return out
class GCAE_CLS_Lite(nn.Module):
    def __init__(self, vec_file, dict_size, wordvec_length,aspect_indexes, num_channels, kernel_sizes, paddings, pool_length, num_aspects, num_classes, device):
        super(GCAE_CLS_Lite, self).__init__()
        self.pool_length = pool_length
        self.aspect_indexes = aspect_indexes
        self.num_aspects = num_aspects
        if vec_file is not None:
            self.wordvec = Vectors(name=vec_file)
            self.embedding = nn.Embedding(300000,300,device = device).from_pretrained(self.wordvec.vectors,freeze=False)
        else:
            self.embedding = nn.Embedding(dict_size,wordvec_length,device = device)
        self.wordConv2 = nn.ModuleList()
        for c, k, p in zip(num_channels, kernel_sizes, paddings):
            self.wordConv1.append(nn.Conv1d(wordvec_length, c, k, padding=p))
            self.wordConv2.append(nn.Conv1d(wordvec_length, c, k, padding=p))
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(self.pool_length)
        self.classifyFC = nn.Linear(sum(num_channels)*self.pool_length,num_classes)
        self.aspectIndexes = aspect_indexes
    def forward(self, x, aspect_label=None):
        input = self.embedding(x).transpose(1,2)
        aspect_vectors = self.embedding(self.aspect_indexes)
        if aspect_label is not None:
            aspect_grad = 1
            # aspect_grad = torch.tanh((aspect_label.to(torch.float)) @ self.aspectFC(aspect_vectors))
            senti_feat = torch.cat([self.maxpool(torch.tanh(conv(input))) for conv in self.wordConv2], dim=1).squeeze(2)
        else:
            senti_feat = torch.cat([self.maxpool(torch.tanh(conv(input))) for conv in self.wordConv2], dim=1).squeeze(2)
        out = self.classifyFC(senti_feat)
        return out
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding2 = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs, dummy=None):
        embeddings = torch.cat((
            self.embedding(inputs), self.embedding2(inputs)), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
class TextCNNReg(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNNReg, self).__init__(**kwargs)
        self.embedding1 = nn.Embedding(vocab_size, embed_size)
        self.embedding2 = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.4)
        self.decoder = nn.Linear(sum(num_channels), 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Sequential(nn.Conv1d(2 * embed_size, c, k),nn.BatchNorm1d()))

    def forward(self, inputs, dummy=None):
        embeddings = torch.cat((
            self.embedding1(inputs), self.embedding2(inputs)), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = torch.clamp(self.decoder(self.dropout(encoding)),1,5)
        return outputs
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 5)
    def forward(self, inputs,dummy=None):
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs
class GCAE_ACSA(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, aspect_indexes, num_classes=3):
        super(GCAE_ACSA, self).__init__()
        self.embedding1 = nn.Embedding(vocab_size, embed_size)
        self.embedding2 = nn.Embedding(vocab_size, embed_size)
        self.aspectFC = nn.Linear(2 * embed_size,num_channels[0])
        self.sentConv = nn.ModuleList()
        self.aspectConv = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.sentConv.append(nn.Conv1d(2 * embed_size, c, k))
            self.aspectConv.append(nn.Conv1d(2 * embed_size, c, k))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.classifyFC = nn.Linear(sum(num_channels),num_classes)
        self.aspectIndexes = aspect_indexes
    def forward(self, x, key=None):
        embeddings = torch.cat((self.embedding1(x), self.embedding2(x)), dim=2).transpose(1,2)
        sent_feat = torch.cat([torch.squeeze(torch.tanh(self.pool(conv(embeddings))), dim=-1)
            for conv in self.sentConv], dim=1)
        if key is not None:
            aspect_embed = torch.cat((self.embedding1(key), self.embedding2(key)), dim=1)
            aspect_feat = torch.cat([self.relu(torch.squeeze(self.pool(conv(embeddings)), dim=-1)+self.aspectFC(aspect_embed)) for conv in self.aspectConv], dim=1)
            out = sent_feat * aspect_feat
        else:
            out = sent_feat
        out = self.classifyFC(self.dropout(out))
        return out
def init_weights(item):
    if isinstance(item,nn.Conv1d) or isinstance(item,nn.Linear):
        torch.nn.init.xavier_uniform_(item.weight)
        item.bias.data.fill_(0.01)
    if type(item) == nn.LSTM:
        for param in item._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(item._parameters[param])