# Representation learning based on multivariate datasets with large missing rates: Teach your model MANNERS

## About

**MANNERS** (*Missing Adjusted Normalization and Nullity Encoding Representation Strategy*) is a novel strategy of (a) missing mask encoding, (b) loss masking and (c) loss normalization via macro avergaing. MANNERS is an approach for reconstruction-based representation learning when faced with large quantities of missing data in multivariate datasets. The idea is to not solely rely on missing data imputation but instead leverage missing data as a potential resource of information and incorporate it into the training pipeline.

The missing mask is a binary tensor indicating observed data points. Mask encoding can be achieved via adding it to the reconstruction task, e.g., with binary cross-entropy (as done in this repo). The mask can be stacked on top of the input or given as condition. The MANNERS loss masking (b) and normalization (c) takes an elementwise loss and the missing mask as input. The loss is mask at missing data points and than averaged across observed data points per variable. Finally a weighted sum of variable losses is calculated for the total loss. Here, rebalancing is applied, such that each variable contributes equally to the total loss. This way, signal from sparse variables is not lost in the training objective. By loss masking and encoding the missing mask, models can learn patterns of missingness that benefit the training objective or downstream task performance.

The MANNERS loss code as a PyTorch module can be found at [src/manners_loss.py](src/manners_loss.py) and is stated below

```
import torch

class MANNERSLoss(torch.nn.Module):
    """
    Computes MANNERS (Missing Adjusted Normalization and Nullity Encoding Representation Strategy) loss. An element-wise loss is masked for missing values. 
    If `mode` is 'micro', the average loss is taken across all observed data points. If `mode` is 'macro', the average loss is taken per channel. 
    Afterwards, each channel contributes equally to the total loss, regardless of the number of observed data points in that channel. This way, 
    the loss is not biased towards channels with more observed data points. 'macro' mode is the recommended setting.

    Args:
        mode (str, optional): The mode of normalization. Can be either 'micro' for averaging across all observed data points, or 'macro' for averaging per channel.
         Defaults to 'macro'.
    """
    def __init__(self, mode: str = 'macro'):
        super(MANNERSLoss, self).__init__()
        available_modes = ['micro', 'macro']
        if mode not in available_modes:
            raise ValueError(f"Mode '{mode}' is not supported. Available modes are {available_modes}")
        self.mode = mode

    def forward(self, elementwise_loss: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """
        `elementwise_loss` tensor and a `missing_mask` must be of shape (N,C,*), where N is the number of samples, C is the number of channels,
        and * represents any additional dimensions (could be none). The `missing_mask` tensor must have the same shape as `elementwise_loss` and contain booleans or ones and zeros.
        ``True`` or a one must indicate a observed data point.

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
        if self.mode == 'micro':
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
```

## Reproducibility

With the rest of this repository the results of our publication **Representation learning based on multivariate datasets with large missing rates: Teach your model MANNERS** can be reproduced.

### Prerequisites

- Clone this repository and install all [requirements](requirements.txt)
- Download the [MIMIC-IV v2.2 database](https://physionet.org/content/mimiciv/2.2/) and unzip the downloaded file. Adjust the constant variable `MIMIC_IV_DATA_DIR` in [src/config.py](src/config.py) to the unzipped directory
- Install [PostgreSQL](https://www.postgresql.org/docs/current/tutorial-install.html) and start a database. Set all PostGreSQL related variables in [src/config.py](src/config.py) accordingly

### Run Experiments

- Create relevant tables of the MIMIC database in PostgreSQL and extract hypotension dataset with `python dataset_creation/create_dataset.py`
- Run Monte-Carlo cross validation using 20 splits with `python experiments/cross_validation_experiments.py --cv_name test_cv --nof_splits 20 --logger json`
- Analyze cross validation results comparing all four training configurations with `python analysis/analyze_cross_validation.py --cv_name test_cv --to_compare manners,mice, manners,fixed, vanilla,mice, vanilla,fixed`
- Optionally, you can train and evaluate individual models with `python experiments/experiment_pipeline`. See [src/arg_parser.py](src/arg_parser.py) for all the different parameterization options 
