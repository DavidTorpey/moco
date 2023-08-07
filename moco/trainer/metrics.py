import torch
import torch.nn.functional as F

from moco.cfg import Config


def info_nce_loss(z1, z2, config: Config):
    features = torch.cat([z1, z2], dim=0)
    labels = torch.cat([torch.arange(config.optim.batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(config.optim.device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(config.optim.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(config.optim.device)

    logits = logits / 0.5

    return logits, labels


def contrastive_accuracy(z1, z2, config: Config):
    logits, labels = info_nce_loss(z1, z2, config)

    return float(torch.mean(
        torch.eq(
            labels, torch.argmax(logits, dim=1)
        ).type(torch.FloatTensor)
    ))
