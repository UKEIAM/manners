import torch
from src.metrics import EvalMetrics
from src.config import CLASS_FEATURES
from src.data_structures import ResultBatch, TargetDataBatch

class CombinedLoss(torch.nn.Module):
    """
    The combined training objective function for the model, which includes discrete reconstruction loss,
    missing data loss, ordinal classification loss, and KL divergence for generative autoencoders. The final loss is a weighted sum of these components.

    Args:
        missing_weight (float, optional): Weight for the missing data loss component. Defaults to 1.0.
        discrete_weight (float, optional): Weight for the discrete data loss component. Defaults to 5.0.
        class_weight (float, optional): Weight for the classification loss component. Defaults to 0.5.
        huber_delta (float, optional): Delta parameter for the Huber loss function. Defaults to 1.0.
        gamma (float, optional): Scaling factor for KL divergence. Defaults to 1.0.
        model_type (str, optional): Type of the model ('discriminative' or 'generative'). Defaults to 'discriminative'.
        normalization (str, optional): Method of normalization to be applied. Defaults to 'manners-macro'.
        encode_mask (bool, optional): Whether the model encodes the missing mask. If True, cross entropy for the missing mask is added to the total loss Defaults to True.
    """
    def __init__(self, missing_weight: float = 1.0, discrete_weight: float = 5.0, class_weight: float = 0.5, huber_delta: float = 1.0,
                 gamma: float = 1.0, model_type: str = 'discriminative', normalization: str = 'manners-macro', encode_mask: bool = True) -> None:
        super().__init__()
        self.missing_weight = missing_weight
        self.discrete_weight = discrete_weight
        self.class_weight = class_weight
        self.huber_delta = huber_delta
        self.gamma = gamma
        model_types = ['discriminative', 'generative']
        if model_type not in model_types:
            raise ValueError(f"Model type '{model_type}' is not supported. Available types are {model_types}")
        self.model_type = model_type
        self.normalization = normalization
        self.encode_mask = encode_mask

    def forward(self, model_result: ResultBatch, target: TargetDataBatch):
        disc_recon_loss = EvalMetrics.calc_huber_loss(model_result, target, self.huber_delta, self.normalization)

        nof_class_features = len(CLASS_FEATURES)
        ordinal_losses = []

        for idx in range(nof_class_features):
            feature_ordinal_loss = EvalMetrics.calc_ordinal_ce_loss(model_result, target, idx, self.normalization)
            ordinal_losses.append(feature_ordinal_loss)

        class_recon_loss = EvalMetrics.combine_ordinal_losses(ordinal_losses, target, self.normalization)

        combined_recon_loss = disc_recon_loss * self.discrete_weight + class_recon_loss * self.class_weight
    
        loss_dict = {
            'disc_reconstruction_loss': disc_recon_loss.item(),
            'class_ordinal_loss': class_recon_loss.item(),
            'combined_recon_loss': combined_recon_loss.item()
        }

        if self.encode_mask:
            missing_loss = EvalMetrics.calc_missing_bce_loss(model_result, target)
            combined_recon_loss += missing_loss * self.missing_weight
            loss_dict['missing_loss'] = missing_loss.item()

        if self.model_type == 'discriminative':
            return combined_recon_loss, loss_dict

        kl_div = EvalMetrics.calc_kl_divergence_loss(model_result)

        combined_loss = combined_recon_loss + self.gamma * kl_div

        loss_dict['kl_divergence'] = kl_div.item()
        loss_dict['combined_loss'] = combined_loss.item()

        return combined_loss, loss_dict