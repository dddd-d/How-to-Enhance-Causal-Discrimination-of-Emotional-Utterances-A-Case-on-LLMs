import os
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import Seq2SeqTrainer, Trainer

"""T5 Multi-Task by Task Prefix
"""
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        #对 features_df 进行列筛选，去掉 ['aux_labels', 'expl_input_ids', 'expl_attention_mask'] 这三列, "~"非
        #to_dict('records')并将剩余的数据转换为字典列表（list of dictionaries）
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids', 'expl_attention_mask': 'attention_mask'}).to_dict('records')

        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)

        return {
            'pred': pred_features,
            'expl': expl_features,
        }

class TaskPrefixDataCollator2(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids', 'expl_attention_mask': 'attention_mask'}).to_dict('records')

        pred_features = super().__call__(pred_features)
        expl_features = super().__call__(expl_features)

        return {
            'pred': pred_features,
            'expl': expl_features,
        }

class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pred_outputs = model(**inputs['pred']) #里面带有label
        expl_outputs = model(**inputs['expl'])

        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)

        loss = self.alpha * pred_outputs[0]  + (1 - self.alpha) * expl_outputs[0]

        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )
    
class TaskPrefixTrainer2(Trainer):
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if 'pred' not in inputs:
            outputs = model(**inputs)
            loss = outputs.loss

            torch.cuda.empty_cache()
            return (loss, outputs) if return_outputs else loss
        else:
            pred_outputs = model(**inputs['pred']) #里面带有label
            expl_outputs = model(**inputs['expl'])

            loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss
            
            torch.cuda.empty_cache()
            return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # 兼容验证阶段的数据格式
        if "pred" not in inputs:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)

        loss = self.alpha * pred_outputs[0]  + (1 - self.alpha) * expl_outputs[0]

        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )

