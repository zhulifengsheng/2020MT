import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable

from torchtext import data, datasets
import spacy

from model import *

import time
import tqdm
import random
import pandas as pd

# dataset
def Data(corpus):
    random.shuffle(corpus)
    print("Reading Lines...")
    input_lang, output_lang = [], []
    for parallel in corpus:
        source, target = parallel[:-1].split('\t')  #用tab来分割
        #print(source)
        #print(target)
        #break
        source = source.strip()
        target = target.strip()
        if source == "" or target == "":
            continue
        input_lang.append(source)
        output_lang.append(target)
    print("Reading Over!") 
    return input_lang, output_lang

def DatasetSplit(corpus, input_lang, output_lang):
    train_list = corpus[:-3756]
    valid_list = corpus[-1622:]
    test_list = corpus[-3756:-1622]
    c_train = {'src':input_lang[:-3756],'trg':output_lang[:-3756]}
    train_df = pd.DataFrame(c_train)
    c_valid = {'src':input_lang[-1622:],'trg':output_lang[-1622:]}
    valid_df = pd.DataFrame(c_valid)
    c_test = {'src':input_lang[-3756:-1622],'trg':output_lang[-3756:-1622]}
    test_df = pd.DataFrame(c_test)
    return train_df, valid_df, test_df

# GetDataset构造并返回Dataset所需的examples和fields
def GetDataset(csv_data, text_field, label_field, test=False):
    fields = [('id', None), ('src', text_field), ('trg', label_field)]
    examples = []
    if test:
        for text in csv_data['src']:  # tqdm的作用是添加进度条
            examples.append(data.Example.fromlist([None, text, None], fields))
    else:
        for text, label in zip(csv_data['src'], csv_data['trg']):
            examples.append(data.Example.fromlist([None, text, label], fields))
    return examples, fields


#if __name__ == "__main__":
corpus = open('eng-ita.txt', 'r', encoding='utf-8').readlines()
input_lang, output_lang = Data(corpus)
train_df, valid_df, test_df = DatasetSplit(corpus, input_lang, output_lang)

# 分词
spacy_it = spacy.load('it_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')
def tokenize_it(text):
    return [tok.text for tok in spacy_it.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# 定义FIELD配置信息
# 主要包含以下数据预处理的配置信息，比如指定分词方法，是否转成小写，起始字符，结束字符，补全字符以及词典等等
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

SRC = data.Field(tokenize=tokenize_it, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)

# 得到构建Dataset所需的examples和fields
train_examples, train_fields = GetDataset(train_df, SRC, TGT)
valid_examples, valid_fields = GetDataset(valid_df, SRC, TGT)
test_examples, test_fields = GetDataset(test_df, SRC, None, True)

MAX_LEN = 100
train = data.Dataset(train_examples, train_fields,
    filter_pred=lambda x: len(vars(x)['src'])<= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
valid = data.Dataset(valid_examples, valid_fields,
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
test = data.Dataset(test_examples, test_fields,
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN)

MIN_FREQ = 2 #统计字典时要考虑词频
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
print("dataset finish!")
#print(len(train.examples[0].src), train.examples[0].src)
#print(len(train.examples[0].trg), train.examples[0].trg)

# 构造迭代器
class MyIterator(data.Iterator):
    '''
    本例使用的是动态批量化，即每个batch的批大小是不同的，
    它是以每个batch的token数量作为统一划分标准，也就是说每个batch的token数基本一致，
    这个机制是通过batch_size_fn来实现的，比如batch1[16,20]，batch2是[8,40]，
    两者的批大小是不同的分别是16和8，但是token总数是一样的都是320
    '''
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

BATCH_SIZE = 6000
train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda'), repeat=False,
                        sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)

valid_iter = MyIterator(valid, batch_size=BATCH_SIZE, device=torch.device('cuda'), repeat=False,
                        sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)

print("Iterator finish!")

# MASK
class BatchMask:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            #self.trg和trg是两个变量 self.trg比trg少最后一列 最后一列不是1pad<"blank">就是结束符3</s>
            #self.trg是用来训练model 作为他的decoder输入的 mask是针对的他
            self.trg = trg[:, :-1]
            #self.trg_y指的是除去起始符<s> 因为target的第一列是2<s>,所以这里要去掉
            #self.trg_y相当于后面做loss的时候计算的softmax输出和实际目标值的loss
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

pad_idx = TGT.vocab.stoi["<blank>"]
def batch_mask(pad_idx, batch):
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return BatchMask(src, trg, pad_idx)

print("start train")
USE_CUDA = torch.cuda.is_available()
print_every = 50
plot_every = 100
plot_losses = []

def time_since(t):
    now = time.time()
    s = now - t
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start_epoch = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    plot_loss_total = 0
    plot_tokens_total = 0
    for i, batch in enumerate(data_iter):
        src = batch.src.cuda() if USE_CUDA else batch.src
        trg = batch.trg.cuda() if USE_CUDA else batch.trg
        src_mask = batch.src_mask.cuda() if USE_CUDA else batch.src_mask
        trg_mask = batch.trg_mask.cuda() if USE_CUDA else batch.trg_mask
        model = model.cuda() if USE_CUDA else model
        out = model.forward(src, trg, src_mask, trg_mask)
        trg_y = batch.trg_y.cuda() if USE_CUDA else batch.trg_y
        ntokens = batch.ntokens.cuda() if USE_CUDA else batch.ntokens
        loss = loss_compute(out, trg_y, ntokens)
        total_loss += loss
        plot_loss_total += loss
        total_tokens += ntokens
        plot_tokens_total += ntokens
        tokens += ntokens
        if i % print_every == 1:
            elapsed = time.time() - start_epoch
            print("Epoch Step: %3d   Loss: %10f    time:%8s     Tokens per Sec: %6.0f     Step: %6d      Lr: %0.8f" %
                  (i, loss / ntokens, time_since(start), tokens / elapsed,
                   loss_compute.opt._step if loss_compute.opt is not None else 0,
                   loss_compute.opt._rate if loss_compute.opt is not None else 0))
            tokens = 0
            start_epoch = time.time()
        if i % plot_every == 1:
            plot_loss_avg = plot_loss_total / plot_tokens_total
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            plot_tokens_total = 0
    return total_loss / total_tokens

model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
model_opt = NoamOpt(model.src_embed[0].d_model, 2, 8000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

start = time.time()
for epoch in range(20):
    print('EPOCH',epoch,'--------------------------------------------------------------')
    model.train()
    run_epoch((batch_mask(pad_idx, b) for b in train_iter),
            model,
            SimpleLossCompute(model.generator, criterion, opt=model_opt))
    model.eval()
    loss=run_epoch((batch_mask(pad_idx, b) for b in valid_iter),
            model,
            SimpleLossCompute(model.generator, criterion, opt=None))
    print(loss)

#state = {'model': model.state_dict(), 'optimizer': model_opt, 'epoch': epoch, 'loss': loss,
#         'plot_losses': plot_losses}
#torch.save(state, 'mt_transformer_it&en%02d.pth.tar' % (epoch))

print('train over')

# 测试
print('test')
model.eval()
#checkpoint = torch.load('mt_transformer_it&en00.pth.tar')
#model.cuda()
#model = nn.DataParallel(model)
#model.load_state_dict(checkpoint['model'])

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    #print("src:",src,src.shape)
    #print("ys:",ys,ys.shape)
    #实际利用model.decode里面tgt,tgt是从头开始生成，可看到decode输入的target的序列长度是可变的，out的shape和target一样
    #循环迭代 ys取out里面最可能的值
    #循环到最后 ys是完整序列长度 长度多少呢就是max_len，此时ys相当于tgt输入decode,所以得到的self_attn.attn注意力的shape就变成了[max,max]
    for i in range(max_len-1):
        out = model.decode(memory,src_mask,Variable(ys),Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        #注意这里输出的out是目前的序列长度
        # print("out:",out.shape)
        #计算out的概率分布
        #等同于out[:,-1,:] 只要求out目前序列长度最后一个out值即可，因为每次迭代循环都是预测下一个word，不用序列所有长度的值
        prob = model.generator(out[:, -1])
        # print("out[:,-1]:",out[:, -1].shape)
        # print("prob:",prob.shape,prob)
        #使用贪婪算法 求得out里面最大概率的值的index
        _, next_word = torch.max(prob, dim = 1)
        # print("next_word:",next_word,next_word.shape)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        # print("下一个ys：",ys)
    return ys

#valid_iter是iterator，先变成list，然后random_shuffle
batch_list=[]
for i, batch in enumerate(valid_iter):
  batch_list.append(batch)
random.shuffle(batch_list)

#目的是在valid_iter里面随机输出一个句子进行翻译
for batch in batch_list:
    b=random.randint(1,batch.src.size(0))
    #print(b)
    src = batch.src.transpose(0, 1)[b:b+1]
    trg = batch.trg.transpose(0, 1)[b:b+1]
    #print("src:",src)
    #print("trg:",trg)
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    #print("out:",out.shape)
    #print("out:",out)
    
    #输出的时候遇到</s>结束符就结束了
    print("Source      :", end="\t")
    for i in range(1, batch.src.size(0)):
        sym = SRC.vocab.itos[src[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    
    print("Target      :", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[trg[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()

    print("Translation :", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break

