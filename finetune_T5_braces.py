from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers
from transformers import Adafactor
import os

import json

import torch
import torch.optim as optim
import pickle
import torch.nn as nn
import random
import wandb
from torch.utils.data import Dataset
import argparse
from torch.utils.data import DataLoader

parser=argparse.ArgumentParser()
parser.add_argument('--train_file',type=str,default=None)
parser.add_argument('--dev_file',type=str,default=None)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--model_name',type=str,default='t5-base')
parser.add_argument('--checkpoint',type=str,default=None)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--exp_name',type=str,default=None)
parser.add_argument('--eval_bs', type = int, default = 3)
parser.add_argument('--bs', type = int, default = 5)
parser.add_argument('--backpropagate', type = int, default = 5)
parser.add_argument("--print_every", type = int, default=1000)
parser.add_argument('--eval_every',type=int,default=5)
# parser.add_argument('--store',type=str,default=None)
args=parser.parse_args()
#wandb.init(project='replacements_new_small_cwq',name=args.exp_name)

if os.path.exists(args.exp_name)==False:
    os.mkdir(args.exp_name)

torch.manual_seed(42)

class Data(Dataset):
        def __init__(self, data):
                self.data = data

        def __len__(self):
                return len(self.data)

        def __getitem__(self, idx):
                return self.data[idx][0], self.data[idx][1]


def read_data(filename):
    f = open(filename)
    lines = f.readlines()
    lis = []
    for i,item in enumerate(lines):
        data = json.loads(item)
        if not data['text'] or not data['summary']:
            continue
        input = data['text'].replace('{','OB').replace('}','CB')
        target = data['summary'].replace('{','OB').replace('}','CB'). \
            replace('<http://www.w3.org/2001/XMLSchema#dateTime>','<extra_id_0>'). \
                replace('<http://www.w3.org/2001/XMLSchema#integer>','<extra_id_1>'). \
                    replace('<http://www.w3.org/2001/XMLSchema#gYear>','<extra_id_2>'). \
                        replace('<http://www.w3.org/2001/XMLSchema#date>','<extra_id_3>'). \
                            replace('<http://www.w3.org/2001/XMLSchema#float>','<extra_id_4>'). \
                                replace('<http://www.w3.org/2001/XMLSchema#gYearMonth>','<extra_id_5>'). \
                                replace('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>','<extra_id_0>')

        lis.append([input, target])

    return lis

data_train = Data(read_data(args.train_file))
data_dev = read_data(args.dev_file)[:1000]
# for i in range(len(data_train)):
#     data_train[i][0] = data_train[i][0].replace('<extra_id_','T').replace('>','')
#     data_train[i][1] = data_train[i][1].replace('<extra_id_','T').replace('>','')


# for i in range(len(data_dev)):
#     data_dev[i][0] = data_dev[i][0].replace('<extra_id_','T').replace('>','')
#     data_dev[i][1] = data_dev[i][1].replace('<extra_id_','T').replace('>','')

class Model(nn.Module):
        def __init__(self,model_name):
                super(Model,self).__init__()
                self.model=T5ForConditionalGeneration.from_pretrained(model_name)

        def forward(self,input):
                outputs=self.model(input_ids=input['input_ids'], \
                                           labels=input['labels'], attention_mask=input['attention_mask'],output_hidden_states=True,output_attentions=True)

                return outputs.loss
                

class Train:
        def __init__(self,data,data_val,args):
                self.data=data
                self.dev_data=data_val
                self.args=args

                self.tokenizer=T5Tokenizer.from_pretrained(args.model_name)
                self.model=nn.DataParallel(Model(args.model_name),device_ids=[args.device])
                self.model.to(f'cuda:{self.model.device_ids[0]}')  
               
                self.optimizer=optim.AdamW(self.model.parameters(),lr=0.0015) # 0.00015 for cwq
                self.lr_scheduler=transformers. \
                        get_polynomial_decay_schedule_with_warmup(self.optimizer, 5000, 30000,power=0.5) # 30000 for all except lcq1

                '''self.optimizer=Adafactor(self.model.parameters(),lr=1e-2,eps=(1e-30, 1e-3),clip_threshold=1.0, \
                beta1=0.0,weight_decay=0.0,relative_step=False, \
                scale_parameter=True,warmup_init=False)'''

                self.epochs = 100
                self.print_every=args.print_every
                self.eval_every=args.eval_every
                self.num_gpus=1
                self.eval_bs=args.eval_bs
                self.bs=args.bs
                self.backpropogate=args.backpropagate

                self.train_dataloader = DataLoader(data, batch_size=self.bs, shuffle=True)
                
                self.train()

        def generate_batch(self):
                output=random.sample(self.data,self.bs)
                inp,label=[],[]
                for dat in output:
                        inp.append(dat[0])
                        label.append(dat[1]) #.replace('<extra_id_','T').replace('>',''))

                return inp,label

        def preprocess_function(self,inputs, targets):
                model_inputs=self.tokenizer(inputs, padding=True, \
                                            return_tensors='pt',max_length=512, truncation=True)
                labels=self.tokenizer(targets,padding=True,max_length=512, truncation=True)

                if True:
                    labels["input_ids"] = [
                        [(l if l != self.tokenizer.pad_token_id else -100) \
                         for l in label] for label in labels["input_ids"]
                    ]
                labels['input_ids']=torch.tensor(labels['input_ids'])
                model_inputs["labels"]=labels["input_ids"].to(f'cuda:{self.model.device_ids[0]}')
                model_inputs["input_ids"]=model_inputs["input_ids"].to(f'cuda:{self.model.device_ids[0]}')
                model_inputs["attention_mask"]=model_inputs["attention_mask"].to(f'cuda:{self.model.device_ids[0]}')

                return model_inputs

        def val(self,o):
                self.model.eval()
                acc,bs,i=0,self.eval_bs,0
                saver=[]
               
                while i<len(self.dev_data):
                    bs_=min(bs,len(self.dev_data)-i)
                    i+=bs_
                    inp,label=[],[]
                    for j in range(i-bs_,i):
                            inp.append(self.dev_data[j][0])
                            label.append(self.dev_data[j][1]) #.replace('<extra_id_','T').replace('>',''))

                    input=self.preprocess_function(inp,label)
                    
                    output=self.model.module.model.generate(input_ids=input['input_ids'],
                                          num_beams=10,attention_mask=input['attention_mask'], \
                                            early_stopping=True, max_length=200,output_hidden_states=True,output_attentions=True)
                    
                    out=self.tokenizer.batch_decode(output,skip_special_tokens=False)

                    for k in range(len(out)):
                            #print(out[k].replace('<pad>','').replace('</s>','').strip())
                            a1=out[k].replace('<pad>','').replace('</s>','').replace('<unk>','').replace('<s>','').strip().replace(' ','')
                            a2=label[k].strip().replace(' ','')
                            #print(a1, '       ', a2)
                            saver.append({'input':inp[k],'gold':label[k].strip(),'generated':out[k].replace('<pad>',''). \
                                          replace('</s>','').replace('<unk>','').replace('<s>','').strip()})
                            if a1==a2:
                                    acc+=1; #print('ttt')
                
                file=open(self.args.exp_name+'/'+str(o)+'dev_result.json','w')
                json.dump(saver,file)
                file.close()

                wandb.log({"epochs": o, "matches": acc})

                return acc

        def train(self):
                loss, j, tot_loss, log_loss, prev_acc = 0, 0, 0, 0, 0
                for epoch in range(self.epochs):
                        for inp, label in self.train_dataloader:
                                self.model.train()
                                input = self.preprocess_function(inp,label)
                                loss_temp = self.model(input)

                                tot_loss += loss_temp.item()
                                loss += loss_temp/self.backpropagate
                                if(j+1)%self.print_every==0:
                                        print('epoch = {}, iteration = {}, training loss = {}'.format(epoch, j, (tot_loss-log_loss)/self.print_every))
                                        log_loss = tot_loss
                                
                                if (j+1)%self.backpropagate == 0:
                                    loss.backward()
                                    self.optimizer.step()
                                    self.lr_scheduler.step()
                                    self.optimizer.zero_grad()
                                    loss = 0

                                j+=1

                        if(epoch+1)%self.eval_every == 0:
                                acc=self.val(epoch)
                                print('validation acc={}'.format(acc))
                                if prev_acc <= acc:
                                    prev_acc = acc
                                    torch.save(self.model.state_dict(),self.args.exp_name+'/'+'checkpoint.pth')

trainer=Train(data_train,data_dev,args)
