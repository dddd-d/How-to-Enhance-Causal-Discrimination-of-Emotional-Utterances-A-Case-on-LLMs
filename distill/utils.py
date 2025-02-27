import numpy as np
import os
import shutil
import logging

from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer, Trainer, TrainingArguments
from transformers import T5ForConditionalGeneration, GPT2LMHeadModel, AutoModelForCausalLM
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import set_seed

from model import TaskPrefixDataCollator, TaskPrefixTrainer, TaskPrefixDataCollator2, TaskPrefixTrainer2
from transformers import DataCollatorForSeq2Seq

def compute_text_acc(preds, labels):
    return np.mean(np.array(preds) == np.array(labels))


def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        predictions1 = np.where(predictions[1] != -100, predictions[1], tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions1, skip_special_tokens=True)
        print(decoded_preds[:5])

        predictions0 = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions0, skip_special_tokens=True)
        print(decoded_preds)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(decoded_labels)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics

def compute_metrics_text2(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        generated_expl = np.argmax(predictions[1][:,:-1,:], axis=-1)
        decoded_preds = tokenizer.batch_decode(generated_expl, skip_special_tokens=True)
        print(decoded_preds[:2])

        generated_pred = np.argmax(predictions[0][:,:-1,:], axis=-1)
        decoded_preds = tokenizer.batch_decode(generated_pred, skip_special_tokens=True)
        print(decoded_preds[:2])

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels[:,1:], skip_special_tokens=True)
        print(decoded_labels[:2])

        #acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': 0}

    return compute_metrics

def compute_metrics_text_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        print(decoded_preds)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics

def get_config_dir(args):
    return f'{args.from_pretrained}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(run)

    if args.from_pretrained == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(args.from_pretrained)
    elif  args.from_pretrained == 'gemma2-2b-it':
        model = AutoModelForCausalLM.from_pretrained(args.from_pretrained)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)

    if args.parallelize:
        model.parallelize()
    
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'  # for model ckpts
    logging_dir = f'logs/{config_dir}/{run}'  # for training logs

    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # clear output dir if already exists
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    if args.from_pretrained == 'gpt2' or 'gemma2-2b-it':
        training_args = TrainingArguments(
            output_dir,
            remove_unused_columns = False,
            evaluation_strategy = 'steps',
            eval_steps=args.eval_steps,
            save_strategy='no',
            logging_dir=logging_dir,
            logging_strategy=logging_strategy,
            logging_steps=args.eval_steps,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            gradient_accumulation_steps=args.grad_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            seed=run,
            local_rank=args.local_rank,
            bf16=args.bf16,
            prediction_loss_only=False,
            # load_best_model_at_end=True,
            # metric_for_best_model='eval_loss',
            # greater_is_better=False
        )

        data_collator = TaskPrefixDataCollator2(tokenizer=tokenizer, mlm=False)

        trainer_kwargs = {
            'alpha': args.alpha,
            'output_rationale': args.output_rationale,
            'model': model,
            'args': training_args,
            'train_dataset': tokenized_datasets["train"],
            'eval_dataset': tokenized_datasets["test"],
            'data_collator': data_collator,
            'processing_class': tokenizer,
            'compute_metrics': compute_metrics,
            #'callbacks': [EarlyStoppingCallback(early_stopping_patience=2)],
        }
        # Trainer 适用于 GPT-2 训练
        trainer = TaskPrefixTrainer2(**trainer_kwargs)

    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir,
            remove_unused_columns = False,
            evaluation_strategy = 'steps',
            eval_steps=args.eval_steps,
            save_strategy='no',
            save_steps=args.eval_steps,
            logging_dir=logging_dir,
            logging_strategy=logging_strategy,
            logging_steps=args.eval_steps,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            gradient_accumulation_steps=args.grad_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            seed=run,
            local_rank=args.local_rank,
            bf16=args.bf16,
            generation_max_length=args.gen_max_len,
            prediction_loss_only=False,
        )

        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)

        trainer_kwargs = {
            'alpha': args.alpha,
            'output_rationale': args.output_rationale,
            'model': model,
            'args': training_args,
            'train_dataset': tokenized_datasets["train"],
            'eval_dataset': tokenized_datasets["test"],
            'data_collator': data_collator,
            'processing_class': tokenizer,
            'compute_metrics': compute_metrics,
        }
        
        trainer = TaskPrefixTrainer(**trainer_kwargs)

    trainer.train()
    # 保存模型
    trainer.save_model()

