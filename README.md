# sparql-vocab-substitution
Repository for ACL 2023 findings short paper titled "The Role of Output Vocabulary in T2T LMs for SPARQL Semantic Parsing"
<hr>
<h3>Files:</h3>
<hr>

```
Train_T5.py                   # Training Prefix tuning experiments
Train_T5_braces.py            # Training Prefix tuning experiments in the "original" setting
Test_T5.py                    # Testing Prefix tuning experiments
Test_T5_braces.py             # Testing Prefix tuning experiments in the "original" setting
finetune_T5.py                # Training Finetuning experiments
finetune_T5_braces.py         # Training Finetuning experiments in the "original" setting
finetune_test_T5.py           # Testing Finetuning experiments
finetune_test_T5_braces.py.   # Testing Finetuning experiments in the "original" setting
```
<hr>
<h4>To run Prefix Tuning experiments:</h4>
First create a virtual environment and install transformers and OpenPrompt

```
virtualenv --python=python3 env
source env/bin/activate
cd transformers
pip install -e ./
cd ../OpenPrompt
python setup.py install
```

To prefix tune T5 models, run the following:

```
# t5-small

CUDA_VISIBLE_DEVICES=0 python Train_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json21.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json21.json --model_name t5-small --num_token 50 --exp_name gqa_t5small_replace_2_1_lr 

CUDA_VISIBLE_DEVICES=0 python Train_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json11.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json11.json --model_name t5-small --num_token 50 --exp_name gqa_t5small_replace_1_1_lr

CUDA_VISIBLE_DEVICES=0 python Train_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json41.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json41.json --model_name t5-small --num_token 50 --exp_name gqa_t5small_replace_4_1_lr

CUDA_VISIBLE_DEVICES=0 python Train_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json81.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json81.json --model_name t5-small --num_token 50 --exp_name gqa_t5small_replace_8_1_lr

CUDA_VISIBLE_DEVICES=0 python Train_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.jsonGW.json --dev_file replacements/dev_grailqa_semparse_jsonlines.jsonGW.json --model_name t5-small --num_token 50 --exp_name gqa_t5small_replace_GW_lr

CUDA_VISIBLE_DEVICES=0 python Train_T5_braces.py --train_file train_grailqa_semparse_jsonlines.json --dev_file dev_grailqa_semparse_jsonlines.json --model_name t5-small --num_token 50 --exp_name gqa_t5small_no_replace_lr

# t5-base

CUDA_VISIBLE_DEVICES=0 python Train_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json21.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json21.json --model_name t5-base --num_token 50 --exp_name gqa_t5base_replace_2_1_lr

CUDA_VISIBLE_DEVICES=0 python Train_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json11.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json11.json --model_name t5-base --num_token 50 --exp_name gqa_t5base_replace_1_1_lr

CUDA_VISIBLE_DEVICES=0 python Train_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json41.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json41.json --model_name t5-base --num_token 50 --exp_name gqa_t5base_replace_4_1_lr

CUDA_VISIBLE_DEVICES=0 python Train_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json81.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json81.json --model_name t5-base --num_token 50 --exp_name gqa_t5base_replace_8_1_lr

CUDA_VISIBLE_DEVICES=0 python Train_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.jsonGW.json --dev_file replacements/dev_grailqa_semparse_jsonlines.jsonGW.json --model_name t5-base --num_token 50 --exp_name gqa_t5base_replace_GW_lr

CUDA_VISIBLE_DEVICES=0 python Train_T5_braces.py --train_file train_grailqa_semparse_jsonlines.json --dev_file dev_grailqa_semparse_jsonlines.json --model_name t5-base --num_token 50 --exp_name gqa_t5base_no_replace_lr
```

To test the prefix tuned models, run the following

```
# t5-small

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json11.json --model_name t5-small --num_token 50 --checkpoint gqa_t5small_replace_1_1_lr/checkpoint.pth --exp_name gqa_t5small_replace_1_1_lr

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json21.json --model_name t5-small --num_token 50 --checkpoint gqa_t5small_replace_2_1_lr/checkpoint.pth --exp_name gqa_t5small_replace_2_1_lr

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json41.json --model_name t5-small --num_token 50 --checkpoint gqa_t5small_replace_4_1_lr/checkpoint.pth --exp_name gqa_t5small_replace_4_1_lr

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json81.json --model_name t5-small --num_token 50 --checkpoint gqa_t5small_replace_8_1_lr/checkpoint.pth --exp_name gqa_t5small_replace_8_1_lr

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.jsonGW.json --model_name t5-small --num_token 50 --checkpoint gqa_t5small_replace_GW_lr/checkpoint.pth --exp_name gqa_t5small_replace_GW_lr

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file test_grailqa_semparse_jsonlines.json --model_name t5-small --num_token 50 --checkpoint gqa_t5small_no_replace_lr/checkpoint.pth --exp_name gqa_t5small_no_replace_lr

# t5-base

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json81.json --model_name t5-base --num_token 50 --checkpoint gqa_t5base_replace_8_1_lr/checkpoint.pth --exp_name gqa_t5base_replace_8_1_lr

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json11.json --model_name t5-base --num_token 50 --checkpoint gqa_t5base_replace_1_1_lr/checkpoint.pth --exp_name gqa_t5base_replace_1_1_lr

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json21.json --model_name t5-base --num_token 50 --checkpoint gqa_t5base_replace_2_1_lr/checkpoint.pth --exp_name gqa_t5base_replace_2_1_lr

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json41.json --model_name t5-base --num_token 50 --checkpoint gqa_t5base_replace_4_1_lr/checkpoint.pth --exp_name gqa_t5base_replace_4_1_lr

CUDA_VISIBLE_DEVICES=0 python Test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.jsonGW.json --model_name t5-base --num_token 50 --checkpoint gqa_t5base_replace_GW_lr/checkpoint.pth --exp_name gqa_t5base_replace_GW_lr

CUDA_VISIBLE_DEVICES=0 python Test_T5_braces.py --test_file test_grailqa_semparse_jsonlines.json --model_name t5-base --num_token 50 --checkpoint gqa_t5base_no_replace_lr/checkpoint.pth --exp_name gqa_t5base_no_replace_lr
```
<hr>

<h4>To run Finetuning experiments:</h4>
First create a virtual environment and install transformers

```
virtualenv --python=python3 env1
source env1/bin/activate
cd transformers1
pip install -e ./
```

To finetune T5 models, run the following:

```
# t5-small

CUDA_VISIBLE_DEVICES=0 python finetune_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.jsonGW.json --dev_file replacements/dev_grailqa_semparse_jsonlines.jsonGW.json --model_name t5-small --exp_name gqa_finetune_t5small_replace_GW_lr

CUDA_VISIBLE_DEVICES=0 python finetune_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json21.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json21.json --model_name t5-small --exp_name gqa_finetune_t5small_replace_2_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json41.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json41.json --model_name t5-small --exp_name gqa_finetune_t5small_replace_4_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json11.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json11.json --model_name t5-small --exp_name gqa_finetune_t5small_replace_1_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json81.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json81.json --model_name t5-small --exp_name gqa_finetune_t5small_replace_8_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_T5_braces.py --train_file train_grailqa_semparse_jsonlines.json --dev_file dev_grailqa_semparse_jsonlines.json --model_name t5-small --exp_name gqa_finetune_t5small_no_replace_lr

# t5-base

CUDA_VISIBLE_DEVICES=0 python finetune_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json21.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json21.json --model_name t5-base --exp_name gqa_finetune_t5base_replace_2_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json11.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json11.json --model_name t5-base --exp_name gqa_finetune_t5base_replace_1_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json41.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json41.json --model_name t5-base --exp_name gqa_finetune_t5base_replace_4_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.json81.json --dev_file replacements/dev_grailqa_semparse_jsonlines.json81.json --model_name t5-base --exp_name gqa_finetune_t5base_replace_8_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_T5.py --train_file replacements/train_grailqa_semparse_jsonlines.jsonGW.json --dev_file replacements/dev_grailqa_semparse_jsonlines.jsonGW.json --model_name t5-base --exp_name gqa_finetune_t5base_replace_GW_lr

CUDA_VISIBLE_DEVICES=0 python finetune_T5_braces.py --train_file train_grailqa_semparse_jsonlines.json --dev_file dev_grailqa_semparse_jsonlines.json --model_name t5-base --exp_name gqa_finetune_t5base_no_replace_lr
```

To test the finetuned models, run the following

```
# t5-small

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.jsonGW.json --model_name t5-small --checkpoint gqa_finetune_t5small_replace_GW_lr/checkpoint.pth --exp_name gqa_finetune_t5small_replace_GW_lr

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json11.json --model_name t5-small --checkpoint gqa_finetune_t5small_replace_1_1_lr/checkpoint.pth --exp_name gqa_finetune_t5small_replace_1_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json21.json --model_name t5-small --checkpoint gqa_finetune_t5small_replace_2_1_lr/checkpoint.pth --exp_name gqa_finetune_t5small_replace_2_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json41.json --model_name t5-small --checkpoint gqa_finetune_t5small_replace_4_1_lr/checkpoint.pth --exp_name gqa_finetune_t5small_replace_4_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json81.json --model_name t5-small --checkpoint gqa_finetune_t5small_replace_8_1_lr/checkpoint.pth --exp_name gqa_finetune_t5small_replace_8_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5_braces.py --test_file test_grailqa_semparse_jsonlines.json --model_name t5-small --checkpoint gqa_finetune_t5small_no_replace_lr/checkpoint.pth --exp_name gqa_finetune_t5small_no_replace_lr

# t5-base

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.jsonGW.json --model_name t5-base --checkpoint gqa_finetune_t5base_replace_GW_lr/checkpoint.pth --exp_name gqa_finetune_t5base_replace_GW_lr

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json11.json --model_name t5-base --checkpoint gqa_finetune_t5base_replace_1_1_lr/checkpoint.pth --exp_name gqa_finetune_t5base_replace_1_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json21.json --model_name t5-base --heckpoint gqa_finetune_t5base_replace_2_1_lr/checkpoint.pth --exp_name gqa_finetune_t5base_replace_2_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json41.json --model_name t5-base --checkpoint gqa_finetune_t5base_replace_4_1_lr/checkpoint.pth --exp_name gqa_finetune_t5base_replace_4_1_lr

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file replacements/test_grailqa_semparse_jsonlines.json81.json --model_name t5-base --checkpoint gqa_finetune_t5base_replace_8_1_lr/checkpoint.pth --exp_name gqa_finetune_t5base_replace_8_1_lr

CUDA_VISIBLE_DEVICES= python finetune_test_T5_braces.py --test_file test_grailqa_semparse_jsonlines.json --model_name t5-base --checkpoint gqa_finetune_t5base_no_replace_lr/checkpoint.pth --exp_name gqa_finetune_t5base_no_replace_lr
```
<hr>
