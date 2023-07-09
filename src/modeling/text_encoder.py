from transformers import AutoTokenizer, AutoModel
from typing import List
import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TextEncoder(torch.nn.Module):

    def __init__(self, model_name: str, max_len: int, extra_tokens: List[str] = None):
        super(TextEncoder, self).__init__()

        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if extra_tokens != None:
            _ = self.tokenizer.add_tokens(extra_tokens, special_tokens=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
 
    def forward(self, sentences):
        
        tokenized = self.tokenizer(sentences, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_len)
        
        att_mask = tokenized["attention_mask"].cuda()
        tokenized = {
            "input_ids":tokenized["input_ids"].cuda(),
            "attention_mask": att_mask
        }
        out = self.encoder(**tokenized)
        
        out = mean_pooling(out, att_mask)
        return out