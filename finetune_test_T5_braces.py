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
parser.add_argument('--test_file',type=str,default=None)
parser.add_argument('--model_name',type=str,default='t5-base')
parser.add_argument('--checkpoint',type=str,default=None)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--exp_name',type=str,default=None)
parser.add_argument('--eval_bs', type = int, default = 2)
# parser.add_argument('--store',type=str,default=None)
args=parser.parse_args()

torch.manual_seed(42)

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

data_dev = read_data(args.test_file)


class Model(nn.Module):
        def __init__(self,model_name):
                super(Model,self).__init__()
                self.model=T5ForConditionalGeneration.from_pretrained(model_name)

        def forward(self,input):
                outputs=self.model(input_ids=input['input_ids'], \
                                           labels=input['labels'], attention_mask=input['attention_mask'],output_hidden_states=True,output_attentions=True)

                return outputs.loss
                

class Train:
        def __init__(self,data_val,args):
                self.dev_data=data_val
                self.args=args

                self.tokenizer=T5Tokenizer.from_pretrained(args.model_name)
                self.model=nn.DataParallel(Model(args.model_name),device_ids=[args.device])
                self.model.to(f'cuda:{self.model.device_ids[0]}')  
                self.model.load_state_dict(torch.load(args.checkpoint))

                self.eval_bs=args.eval_bs
                
                print('testing')
                self.val(0)

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
                                            early_stopping=True, max_length=512,output_hidden_states=True,output_attentions=True)
                    
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
                
                file=open(self.args.exp_name+'/'+str(o)+'test_result.json','w')
                json.dump(saver,file)
                file.close()

                print(acc*100/len(saver))

trainer=Train(data_dev,args)
