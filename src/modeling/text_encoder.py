from transformers import AutoTokenizer, AutoModel
from typing import List
import torch
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TextEncoder(torch.nn.Module):

    def __init__(self, model_name: str, max_len: int, extra_tokens: List[str] = None, end_prior_context: str = "[BFR]", start_future_context: str = "[AFT]"):
        super(TextEncoder, self).__init__()

        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if extra_tokens != None:
            _ = self.tokenizer.add_tokens(extra_tokens, special_tokens=True)
        self.end_prior_context = self.tokenizer.convert_tokens_to_ids(end_prior_context)
        self.start_future_context = self.tokenizer.convert_tokens_to_ids(start_future_context)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
 
    def get_context_mask(self, tokenized):
        context_masks = []
        for x, mask in zip(tokenized["input_ids"].numpy(), tokenized["attention_mask"].numpy()):
            if self.start_future_context in list(x) and self.end_prior_context in list(x):
                i, j = 0, self.max_len - 1
                while x[i] != self.end_prior_context and x[j] != self.start_future_context and i < j:
                    if x[i] != self.end_prior_context:
                        mask[i] = 0.0
                        i += 1
                    if x[j] != self.start_future_context:
                        mask[j] = 0.0
                        j -= 1
                mask[i] = 0.0
                mask[j] = 0.0
            context_masks.append(np.expand_dims(mask, axis=0))
        return torch.Tensor(np.concatenate(context_masks, axis=0))

    def forward(self, sentences):
        
        tokenized = self.tokenizer(sentences, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_len)
        #context_mask = self.get_context_mask(tokenized).cuda()

        att_mask = tokenized["attention_mask"].cuda()
        tokenized = {
            "input_ids":tokenized["input_ids"].cuda(),
            "attention_mask": att_mask
        }
        out = self.encoder(**tokenized)
        #print(context_mask.shape)
        #print(att_mask.shape)
        out = mean_pooling(out, att_mask)
        return out
