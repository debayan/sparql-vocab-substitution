from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers
import os

import json
from transformers import Adafactor
import torch
import torch.optim as optim
import pickle
import torch.nn as nn
import random


from openprompt.plms import load_plm
from openprompt import PromptDataLoader
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt import PromptForGeneration
from openprompt.data_utils.utils import InputExample

import argparse
import wandb

parser=argparse.ArgumentParser()
parser.add_argument('--train_file',type=str,default=None)
parser.add_argument('--dev_file',type=str,default=None)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--model_name',type=str,default='t5-base')
parser.add_argument('--checkpoint',type=str,default=None)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--num_token',type=int,default=512)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument('--exp_name',type=str,default=None)
parser.add_argument("--eval_bs", type = int, default=6)
parser.add_argument('--bs',type=int,default=5)
args=parser.parse_args()
#wandb.init(project='replacements_new_small_cwq',name=args.exp_name)

if os.path.exists(args.exp_name)==False:
    os.mkdir(args.exp_name)

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
                                   
        lis.append(InputExample(guid = str(i),text_a = input, tgt_text = target))

    return lis

dataset={}
dataset['train'] = read_data(args.train_file)
dataset['validation'] = read_data(args.dev_file)[:1000]           

class Train:
        def __init__(self,dataset,args):
                self.dataset = dataset
                self.args=args

                self.epochs = 200
                self.print_every=1000
                self.eval_every=10000
                self.num_gpus=1
                self.eval_bs=args.eval_bs
                self.bs=args.bs
                self.back_propogate=2
                self.use_cuda = True

                plm, tokenizer, model_config, WrapperClass = load_plm(args.model_name.split('-')[0], args.model_name)
                self.mytemplate = PrefixTuningTemplate(model=plm, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'})

                self.train_dataloader = PromptDataLoader(dataset=dataset["train"], template=self.mytemplate, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
                    batch_size=self.bs,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your tempalte doesn't contain one, or you model may fail to stop generation.
                    truncate_method="head")

                self.validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=self.mytemplate, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
                    batch_size=self.eval_bs,shuffle=False, teacher_forcing=False, predict_eos_token=True,
                    truncate_method="head")

                self.prompt_model = PromptForGeneration(plm=plm,template=self.mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)                
               
                if self.use_cuda:
                    self.prompt_model = self.prompt_model.cuda()

                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in self.mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                        "weight_decay": 0.0,
                    },
                ]

                #self.optimizer=optim.AdamW(optimizer_grouped_parameters, lr=0.00005)
                self.optimizer=Adafactor(optimizer_grouped_parameters,lr=1e-3,eps=(1e-30, 1e-3),clip_threshold=1.0, \
                beta1=0.0,weight_decay=0.0,relative_step=False, \
                scale_parameter=True,warmup_init=False)
                #self.lr_scheduler=transformers.get_polynomial_decay_schedule_with_warmup(self.optimizer, 5000, 30000, power = 0.5)
                
                self.train()

        def val(self,epoch):
            generated_sentence = []
            groundtruth_sentence = []
            self.prompt_model.eval()
            
            for step, inputs in enumerate(self.validation_dataloader):
                if self.use_cuda:
                    inputs = inputs.cuda()
                
                _,output_sentence=self.prompt_model.generate(inputs,
                                        num_beams=10, \
                                        early_stopping=True, max_length=200,output_hidden_states=True,output_attentions=True)

                output_sentence=[o.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for o in output_sentence]
                gold = [ii.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for ii in inputs['tgt_text']]
                
                generated_sentence.extend(output_sentence)
                groundtruth_sentence.extend(gold)

            acc = 0
            file=open(self.args.exp_name+'/'+str(epoch)+'gen.txt','w')
            file1=open(self.args.exp_name+'/'+str(epoch)+'ref.txt','w')
            for i in range(len(generated_sentence)):
                file1.write(groundtruth_sentence[i].strip()+'\n')
                file.write(generated_sentence[i].strip()+'\n')
                if groundtruth_sentence[i].strip().replace(' ','') == generated_sentence[i].strip().replace(' ',''): acc+=1

            file.close()
            file1.close()
                
            wandb.log({"epoch": epoch, "matches": acc })
            print(acc)
            return acc

        def train(self):
            global_step = 0
            tot_loss = 0
            log_loss = 0
            prev_acc = 0
            for epoch in range(self.epochs):
                self.prompt_model.train()
                for step, inputs in enumerate(self.train_dataloader):
                    global_step +=1
                    if self.use_cuda:
                        inputs = inputs.cuda()
                    loss = self.prompt_model(inputs)
                    #loss += 0.1*(self.mytemplate.loss + self.mytemplate.loss1)
                    loss.backward()
                    tot_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(self.mytemplate.parameters(), 1.0)
                    self.optimizer.step()
                    #self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    if (1+global_step) %self.print_every ==0:
                        print("Epoch {}, global_step {} average loss: {}".format(epoch, global_step, (tot_loss-log_loss)/self.print_every), flush=True)
                        #print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
                        log_loss = tot_loss
                if (1+epoch) % 5 ==0:
                    acc = self.val(epoch)
                    if prev_acc <= acc:
                        prev_acc =  acc
                        torch.save(self.mytemplate.state_dict(),self.args.exp_name+'/'+'checkpoint.pth')
                        

trainer=Train(dataset,args)

