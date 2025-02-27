import argparse
from transformers import AutoTokenizer
from utils import compute_metrics_text, compute_metrics_text2
from utils import train_and_evaluate
import warnings
import os
warnings.filterwarnings("ignore")
from datasets import DatasetDict,load_dataset 
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
os.environ["WANDB_DISABLED"] = "true"

def process_input(example):
    example['input'] = example['x']
    example['aux_label'] = example['y_c']
    example['label'] = example['y']
    return example
    
def load_data(train_path, test_path):
   
    data_files = {
        'train': train_path,
        'test': test_path,
    }
    datasets = load_dataset('json', data_files=data_files)  
    datasets = datasets.map(process_input, load_from_cache_file=False, remove_columns=['x', 'y', 'y_c', 'type'],)
    
    datasets = DatasetDict({
        'train': datasets['train'],
        'test': datasets['test'],
    })  
     
    return datasets

def run(args):
    #### Prepare datasets
    datasets = load_data(args.train_path, args.test_path)

    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    if args.from_pretrained == 'gpt2':
        # 确保 tokenizer 添加 pad token（GPT-2 默认没有）
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        model_inputs = tokenizer(['Answer the following questions directly:\n' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
        expl_model_inputs = tokenizer(['Think and answer the following questions step by step:\n' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

        label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
        rationale_output_encodings = tokenizer(examples['aux_label'], max_length=1024, truncation=True)

        model_inputs['labels'] = label_output_encodings['input_ids']
        model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

        return model_inputs
    
    def tokenize_function2(examples):
        model_inputs = tokenizer(['Answer the following questions directly:\n' + examples['input'][i] + '\nAnswer: ' + examples['label'][i] for i in range(len(examples['input']))], truncation=True, padding=True, return_tensors='pt',max_length=args.max_input_length)
        expl_model_inputs = tokenizer(['Think and answer the following questions step by step:\n' + examples['input'][i] + '\nAnswer: ' + examples['aux_label'][i] for i in range(len(examples['input']))], truncation=True, padding=True, return_tensors='pt',max_length=args.max_input_length)
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

        model_inputs['labels'] = model_inputs['input_ids']
        model_inputs['aux_labels'] = expl_model_inputs['input_ids']

        return model_inputs
    
    if args.from_pretrained == 'gpt2' or 'gemma2-2b-it':
        tokenized_datasets = datasets.map(
            tokenize_function2,
            remove_columns=['input', 'aux_label', 'label'],
            batched=True,
            load_from_cache_file=False
        )
        compute_metrics = compute_metrics_text2(tokenizer)
    else:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'aux_label', 'label'],
            batched=True,
            load_from_cache_file=False
        )
        compute_metrics = compute_metrics_text(tokenizer)

    print(tokenized_datasets)
    print(type(tokenized_datasets['train']['input_ids']))
    print(tokenized_datasets['train']['input_ids'][0])

    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/train-for-distill2.json')
    parser.add_argument('--test_path', type=str, default='data/test-for-distill2.json')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='gpt2')
    parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--max_input_length', type=int, default=300)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=1024)
    parser.add_argument('--parallelize', type=bool, default=False)
    parser.add_argument('--bf16', type=bool, default=True)
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', type=bool, default=True)

    args = parser.parse_args()

    run(args)