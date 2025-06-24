import torch

class MANNERS(torch.nn.Module):
    """
    Computes MANNERS (Missing Adjusted Normalization for Network Error Reduction Strategy). An element-wise loss is masked for missing values. 
    If `mode` is 'macro', the average loss is taken across all non-missing data points. If `mode` is 'micro', the average loss is taken per channel. 
    Afterwards, each channel contributes equally to the total loss, regardless of the number of non-missing data points in that channel. This way, 
    the loss is not biased towards channels with more non-missing data points. 'micro' mode is used in the publication.

    Args:
        mode (str, optional): The mode of normalization. Can be either 'micro' for averaging per channel, or 'macro' for averaging across all non-missing data points.
         Defaults to 'micro'.
    """
    def __init__(self, mode: str = 'micro'):
        super(MANNERS, self).__init__()
        available_modes = ['micro', 'macro']
        if mode not in available_modes:
            raise ValueError(f"Mode '{mode}' is not supported. Available modes are {available_modes}")
        self.mode = mode

    def forward(self, elementwise_loss: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """
        `elementwise_loss` tensor and a `missing_mask` must be of shape (N,C,*), where N is the number of samples, C is the number of channels,
        and * represents any additional dimensions (could be none). The `missing_mask` tensor must have the same shape as `elementwise_loss` and contain booleans or ones and zeros.
        ``True`` or a one must indicate a non-missing data point.

        Args:
            elementwise_loss (torch.Tensor): The element-wise loss tensor.
            missing_mask (torch.Tensor): The mask tensor indicating which data points are missing.

        Returns:
            torch.Tensor: The total loss function value.
        """
        if elementwise_loss.shape != missing_mask.shape:
            raise ValueError("The shape of 'elementwise_loss' must match the shape of 'missing_mask'")
        if len(elementwise_loss.shape) < 2:
            raise ValueError("'elementwise_loss' must have at least two dimensions")
        if self.mode == 'macro':
            total_loss = (elementwise_loss * missing_mask).sum()
            total_loss = total_loss / missing_mask.sum()
        else:
            missing_adj_loss = elementwise_loss * missing_mask
            denominator = missing_mask.to(int)
            feature_sample_loss = missing_adj_loss
            while len(denominator.shape) > 2:
                denominator = denominator.sum(-1)
                feature_sample_loss = feature_sample_loss.sum(-1)
            has_non_missing = denominator.to(bool)
            total_nof_non_missing = has_non_missing.sum()
            denominator[~has_non_missing] = 1
            feature_sample_loss = feature_sample_loss / denominator
            total_loss = feature_sample_loss.sum() / total_nof_non_missing
        return total_loss
