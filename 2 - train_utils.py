import torch
import pandas as pd
import jieba
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from torchtext.vocab import Vectors
from torch.nn.utils.rnn import pad_sequence


class CrossEntropyLoss_LS(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, label_smooth=None, class_num=5):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num
        self.eps = 1e-12
    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output [batch_size, num_classe]
            target: ground truth of sampler [num_classes]
        '''
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num-1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)
        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + self.eps).sum(dim=1))
        return loss.mean()
class collector():
    def __init__(self, indexes):
        self.data = {}
        for index in indexes:
            self.data[index] = []
    def genDataFrame(self):
        return pd.DataFrame(self.data)
    def genCSV(self, name):
        pd.DataFrame(self.data).to_csv(name,encoding='utf-8',index=False)
    def append(self, item, index):
        self.data[index].append(item)
def evaluateTSV(model, state_dict_path, dataloader, loss_func, device, outputPath=None,join=False):
    model.eval()
    if outputPath is not None:
        f = open(outputPath, "w")
    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path,map_location=device))
    avg_loss = 0
    correct = 0
    total = 0
    if outputPath is not None:
        f.write("index\tprediction\n")
    i = 0
    for X,keyword,y in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device)
        keyword = keyword.to(device)
        output = model(X,keyword)
        test_loss = loss_func(output,y)
        avg_loss += test_loss.item()
        output = output.argmax(1)
        if join:
            output = output // 2.5
        result = output == y
        correct += result.sum().item()
        total += output.shape[0]
        if outputPath is not None:
            if output.shape == 1:
                f.write(f"{i}\t{output.item()-1}\t{y.item()-1}\n")
                i += 1
            else:
                for item in output:
                    f.write(f"{i}\t{item.item()-1}\t{y.item()-1}\n")
                    i += 1


    acc = correct/total*100
    avg_loss /= len(dataloader)
    print(f"Avg Test Loss:{avg_loss:>5.3f}")
    print(f"Accuracy: {acc:<6.2f}%")
    if outputPath is not None:
        f.close()
    model.train()
    return avg_loss,acc
def evaluateSENT(model, state_dict_path, dataloader, loss_func, device, outputPath=None, FullLabel=False, Reg=False):
    model.eval()
    if outputPath is not None:
        f = open(outputPath, "w")
    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path,map_location=device))
    avg_loss = 0
    correct = 0
    total = 0
    if outputPath is not None:
        f.write("index\tprediction\tcorrect\n")
    i = 0
    for X,y in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        if Reg:
            if FullLabel:
                test_loss = loss_func(output.squeeze(1),y[:,0])
            else:
                test_loss = loss_func(output,y)
        else:
            output = torch.round(output)
            if FullLabel:
                test_loss = loss_func(output.squeeze(1), y[:, 0])
                result = output == y[:, 0]
            else:
                test_loss = loss_func(output, y)
                result = output == y
            correct += result.sum().item()
            total += output.shape[0]
        avg_loss += test_loss.item()
        if outputPath is not None:
            if output.shape == 1:
                f.write(f"{i}\t{output.item()}\n")
                # f.write(f"{i}\t{output.item()}\t{y.squeeze(1).item()}\n")
                i += 1
            else:
                for item in output:
                    f.write(f"{i}\t{item.item()}\n")
                    # f.write(f"{i}\t{output.item()}\t{y.squeeze(1).item()}\n")
                    i += 1
    avg_loss /= len(dataloader)
    print(f"Avg Test Loss:{avg_loss:>5.3f}")
    if not Reg:
        acc = correct / total * 100
        print(f"Accuracy: {acc:<6.2f}%")
    if outputPath is not None:
        f.close()
    model.train()
    if Reg:
        return avg_loss
    else:
        return avg_loss,acc
def evaluateSENT_CLS(model, state_dict_path, dataloader, loss_func, device, outputPath=None, FullLabel=False):
    model.eval()
    if outputPath is not None:
        f = open(outputPath, "w")
    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path,map_location=device))
    avg_loss = 0
    correct = 0
    total = 0
    if outputPath is not None:
        f.write("index\tprediction\tlabel\n")
    i = 0
    for X,y in tqdm(dataloader):
        X = X.to(device)
        y = y = (y-1).to(device).to(torch.long)
        output = model(X)
        predict = output.argmax(1)
        if FullLabel:
            test_loss = loss_func(output, y[:, 0])
            result = predict == y[:, 0]
        else:
            test_loss = loss_func(output, y)
            result = predict == y
        correct += result.sum().item()
        total += output.shape[0]
        avg_loss += test_loss.item()
        if outputPath is not None:
            if predict.shape == 1:
                f.write(f"{i}\t{predict.item()}\t{y[0][0].item()}\n")
                i += 1
            else:
                for item in predict:
                    f.write(f"{i}\t{item.item()}\t{y[0][0].item()}\n")
                    i += 1
    avg_loss /= len(dataloader)
    print(f"Avg Test Loss:{avg_loss:>5.3f}")
    acc = correct / total * 100
    print(f"Accuracy: {acc:<6.2f}%")
    if outputPath is not None:
        f.close()
    model.train()
    return avg_loss,acc
def evaluateASPECTForFull(model, state_dict_path, dataloader, loss_func, aspect_indexes, device, outputPath=None):
    model.eval()
    if outputPath is not None:
        f = open(outputPath, "w")
    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path,map_location=device))
    avg_loss = 0
    correct = 0
    total = 0
    if outputPath is not None:
        f.write("index\tprediction\tlabel\n")
    i = 0
    for X,y in tqdm(dataloader):
        X = X.to(device)
        y = (y[:,1:]+1).to(device).to(torch.long)
        for j in range(y.shape[1]):
            if y[0][j] > -1:
                output = model(X,aspect_indexes[j].reshape(1))
                predict = output.argmax(1)
                test_loss = loss_func(output, y[0][j])
                result = predict == y[0][j]
                correct += result.sum().item()
                total += 1
                avg_loss += test_loss.item()
                if outputPath is not None:
                    f.write(f"{i}\t{predict.item()}\t{y[0][j].item()}\n")
                    i += 1
                    # else:
                    #     for item in predict:
                    #         f.write(f"{i}\t{item.item()}\t{y[0][j].item()}\n")
                    #         i += 1
            else:
                continue
    avg_loss /= total
    print(f"Avg Test Loss:{avg_loss:>5.3f}")
    acc = correct / total * 100
    print(f"Accuracy: {acc:<6.2f}%")
    if outputPath is not None:
        f.close()
    model.train()
    return avg_loss,acc
class ASAP_ASPECT_DICT(torch.utils.data.Dataset):
    def __init__(self, data_file, dict_obj, aspect_dict,mode="train"):
        self.raw_data = pd.read_csv(data_file,sep='\t')
        self.vocab = dict_obj
        self.aspect_dict = aspect_dict
        self.label_dict = {"text_a":1,"cate":2,"label":3}
        self.mode = mode

    def __len__(self):
        return len(self.raw_data)
    def padVector(self):
        return 0
    def clean_str(self, string):
        for ch in '!"#$&*+，、-.:；<=>？@[]《》^_|·~‘’':
            string = string.replace(ch,"")
        return string
    def __getitem__(self, idx):
        line = self.raw_data.iloc[idx]
        indexes = [self.label_dict["cate"],self.label_dict["label"],self.label_dict["text_a"]]
        if self.mode == "train":
            for i in range(len(indexes)):
                indexes[i] -= 1
        keyword = self.aspect_dict[line[indexes[0]]]
        label = torch.tensor(line[indexes[1]])
        seg_list = list(jieba.cut(self.clean_str(line[indexes[2]])))
        sentence = []
        for word in seg_list:
            if word in self.vocab.token_to_idx:
                vector_index = self.vocab[word]
                sentence.append(vector_index)
            else:
                # For Unknown words.
                sentence.append(1)
        return torch.as_tensor(sentence), label+1, keyword
class ASAP_WIKI(torch.utils.data.Dataset):
    def __init__(self, data_file, vec_file, max_seq_length,aspect_dict,mode="train"):
        self.raw_data = pd.read_csv(data_file,sep='\t')
        self.wordvec = Vectors(name=vec_file)
        self.max_seq_length = max_seq_length
        self.aspect_dict = aspect_dict
        self.label_dict = {"text_a":1,"cate":2,"label":3}
        self.mode = mode

    def __len__(self):
        return len(self.raw_data)
    def padVector(self):
        return self.wordvec.stoi['。']
    def clean_str(self, string):
        for ch in '!"#$&*+，、-.:；<=>？@[]《》^_|·~‘’':
            string = string.replace(ch,"")
        return string
    def __getitem__(self, idx):
        line = self.raw_data.iloc[idx]
        indexes = [self.label_dict["cate"],self.label_dict["label"],self.label_dict["text_a"]]
        if self.mode == "train":
            for i in range(len(indexes)):
                indexes[i] -= 1
        keyword = self.aspect_dict[line[indexes[0]]]
        label = torch.tensor(line[indexes[1]])
        seg_list = list(jieba.cut(self.clean_str(line[indexes[2]])))
        sentence = []
        for word in seg_list[:min(len(seg_list),self.max_seq_length)]:
            if word in self.wordvec.stoi:
                vector_index = self.wordvec.stoi[word]
                sentence.append(vector_index)
            else:
                sentence.append(0)
        return torch.tensor(sentence), label+1, keyword
class ASAP_Full(torch.utils.data.Dataset):
    def __init__(self, data_file, dictObj):
        self.raw_data = pd.read_csv(data_file)
        self.lookupTable = dictObj;

    def __len__(self):
        return len(self.raw_data)
    def clean_str(self, string):
        for ch in '!"#$&*+，、-.:；<=>？@[]《》^_|·~‘’':
            string = string.replace(ch,"")
        return string
    def __getitem__(self, idx):
        line = self.raw_data.iloc[idx]
        label = torch.tensor(line[2:])
        seg_list = list(jieba.cut(self.clean_str(line[1])))
        sentence = []
        for word in seg_list:
            if word in self.lookupTable.token_to_idx:
                vector_index = self.lookupTable[word]
                sentence.append(vector_index)
            else:
                sentence.append(1)
        return torch.tensor(sentence), label
class ASAP_SENT_DICT(torch.utils.data.Dataset):
    def __init__(self, data_file, dict_obj, mode="train"):
        self.raw_data = pd.read_csv(data_file, sep='\t')
        self.label_dict = {"text_a": 1, "star": 2}
        self.vocab = dict_obj
        self.mode = mode

    def __len__(self):
        return len(self.raw_data)

    def padVector(self):
        return 0

    def clean_str(self, string):
        for ch in '!"#$&*+，、-.:；<=>？@[]《》^_|·~‘’':
            string = string.replace(ch, "")
        return string

    def __getitem__(self, idx):
        line = self.raw_data.iloc[idx]
        indexes = [self.label_dict["text_a"], self.label_dict["star"]]
        if self.mode == "train":
            for i in range(len(indexes)):
                indexes[i] -= 1
        if self.mode == "test":
            label = torch.tensor(0)
        else:
            label = torch.tensor(line[indexes[1]])
        seg_list = list(jieba.cut(self.clean_str(line[indexes[0]])))
        sentence = []
        for word in seg_list:
            if word in self.vocab.token_to_idx:
                vector_index = self.vocab[word]
                sentence.append(vector_index)
            else:
                #For Unknown words.
                sentence.append(1)
        # if len(seg_list)<self.max_seq_length:
        #     sentence = sentence + [self.padVector()]*(self.max_seq_length-len(seg_list))
        return torch.as_tensor(sentence), label
def collcate_pad(data):
    reviews = [item[0] for item in data]
    reviews = torch.as_tensor(pad_sequence(reviews,batch_first=True))
    labels = torch.as_tensor([item[1] for item in data])
    cates = torch.as_tensor([item[2] for item in data])
    return reviews,cates,labels
def collcate_pad_SENT(data):
    reviews = [item[0] for item in data]
    reviews = torch.as_tensor(pad_sequence(reviews, batch_first=True))
    labels = torch.tensor([item[1] for item in data])
    return reviews, labels
def collcate_pad_Full(data):
    reviews = [item[0] for item in data]
    reviews = torch.as_tensor(pad_sequence(reviews, batch_first=True))
    labels = torch.cat([item[1] for item in data]).reshape(len(data),-1)
    return reviews,labels
def get_aspect_vector_DICT(vocabObj):
    aspect_map = {'Location#Transportation':"交通", 'Location#Downtown':"商圈",
                   'Location#Easy_to_find':"定位", 'Service#Queue':"排队", 'Service#Hospitality':"服务",
                   'Service#Parking':"停车", 'Service#Timely':"准时", 'Price#Level':"价格",
                   'Price#Cost_effective':"性价比", 'Price#Discount':"折扣", 'Ambience#Decoration':"装修",
                   'Ambience#Noise':"噪音", 'Ambience#Space':"空间", 'Ambience#Sanitary':"卫生", 'Food#Portion':"分量",
                   'Food#Taste':"口味", 'Food#Appearance':"外观", 'Food#Recommend':"推荐"}
    aspect_indexes = []
    wordvec_dict = {}
    for key in aspect_map:
        indexKey = torch.tensor(vocabObj.token_to_idx[aspect_map[key]])
        aspect_indexes.append(indexKey)
        wordvec_dict[key] = indexKey
    return aspect_indexes,wordvec_dict