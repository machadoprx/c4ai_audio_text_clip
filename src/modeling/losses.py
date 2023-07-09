import torch
from torch import nn
import torch.nn.functional as F

def sup_contrastive_loss(embeddings, targets, temperature=0.5):
    assert embeddings.size(0) == len(targets)
    batch_size = embeddings.size(0)
    
    mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    similarity_matrix = similarity_matrix.masked_fill(mask, 0.0)
    
    targets = targets.contiguous().view(-1, 1)
    targets = torch.eq(targets, targets.T).float()
    targets = targets.masked_fill(mask, 0.0)
    
    loss = - (targets * F.log_softmax(similarity_matrix, dim=-1)).sum(dim=-1) / targets.sum(dim=-1)
    
    return loss.mean()

def unsupervised_contrastive_loss(embeddings1, embeddings2, temperature=0.5, weights=None):
    assert embeddings1.size(0) == embeddings2.size(0)
    batch_size = embeddings1.size(0)
    embeddings = torch.cat([embeddings1, embeddings2], dim=0) # (2xN,f)

    mask = torch.eye(2 * batch_size, device=embeddings.device, dtype=torch.bool)
    
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature # (2xN,2xN)
    similarity_matrix = similarity_matrix.masked_fill(mask, 0.0)
    
    targets = torch.eye(batch_size, device=embeddings.device, dtype=torch.float32)
    targets = targets.repeat(2, 2) # (2xN, 2xN) -> [[I, I], [I, I]]
    targets = targets.masked_fill(mask, 0.0) # (2xN, 2xN) -> [[0, I], [I, 0]]
    
    loss = - (targets * F.log_softmax(similarity_matrix, dim=-1)).sum(dim=-1) # No need to divide by 1
    if weights != None:
        loss = loss * weights

    return loss.mean()