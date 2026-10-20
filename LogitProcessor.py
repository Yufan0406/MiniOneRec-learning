from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count=0 # 记录当前生成的token是第几个，方便后续在prefix_allowed_tokens_fn中取出对应的前缀（每生成一个位置的token，就会调用一次/训练阶段前向传播）
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        '''
        input_ids: (batch_size * num_beams, sequence_length)
        scores: (batch_size * num_beams, vocab_size)
        batch_size实际就是多少条不同的prompt，对应不同的历史行为序列
        '''

        scores = torch.nn.functional.log_softmax(scores, dim=-1) # scores是模型输出的logits，经过log_softmax后变为log概率分布
        mask = torch.full_like(scores, float('-inf')) # 初始化一个和scores形状相同的mask矩阵，初始值为负无穷，用来后续排除不合法的token
            
        for batch_id, beam_sent in enumerate(
            input_ids.view(-1, self._num_beams, input_ids.shape[-1])
            ):
            # 取每一种prompts所有beam的生成结果
            for beam_id, sent in enumerate(beam_sent):
                # 各个beam的回答
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    hash_key=sent[-self.count:]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key) # 实际会调用minionerec_trainer.py中的prefix_allowed_tokens_fn函数，返回当前前缀对应的合法token集合

                if len(prefix_allowed_tokens) == 0:
                    warnings.warn(
                        f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                        f"This indicates the model generated an unexpected token. "
                    )
                    # Force EOS token to end invalid sequence
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0 # 如果没有合法的token，就强制将eos_token_id对应的mask值设为0，保证模型可以结束当前生成序列。相当于告诉模型后续的生成结果不合法，直接结束当前生成序列
                    continue 
                
                mask[batch_id * self._num_beams + beam_id, 
                     prefix_allowed_tokens # 多个合法token的mask重新设置为0
                     ] = 0

        self.count += 1

        scores = scores + mask # 合法token的log概率不变，不合法token的log概率变为负无穷，保证模型只能输出合法token
        return scores