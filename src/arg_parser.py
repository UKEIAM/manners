import argparse

class ExperimentArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        base = self.add_argument_group('base', 'Basic experiment parameters')
        base.add_argument('--pipeline', type=str, nargs='*', help='The steps performed in the pipeline', choices=['train', 'downstream', 'reconstruction', 'synthesis'], required=True)
        base.add_argument('--model_type', type=str, default='discriminative', choices=['discriminative', 'generative'], help='Whether a discriminative or generative model is trained')
        base.add_argument('--name', default=None, help='Name of the model, timestamp will be added when training new model. When omitted a random pet name will be used. Cannot be ommited, if existing model is used (not training new one)')
        base.add_argument('--device', default='cuda', help='Device used for training')
        base.add_argument('--seed', type=int, default=42, help='Seed used for dataset splitting and all other random selections ensuring reproducibility')
        base.add_argument('--debug', action='store_true', help='Flag to run in debug mode')
        base.add_argument('--verbose', action='store_true', help='Flag to print workflow messages')

        train = self.add_argument_group('train', 'Basic training parameters')
        train.add_argument('--iterations', type=int, default=200000, help='Number of max iterations used for training')
        train.add_argument('--learnrate', type=float, default=0.001, help='Initial learnrate')
        train.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay used by the optimizer')
        train.add_argument('--batch_size', type=int, default=64, help='Batch size used for training')
        train.add_argument('--logger', type=str, default='json', choices=['wandb', 'json'], help='Log to WandB or to local JSON file. Disabled if "--debug" is used')
        train.add_argument('--encode_mask', action='store_true', help='Wether to encode the missing data mask along with the data. Will also be decoded and reconstruction loss calculation')
        train.add_argument('--imputation_mode', type=str, default='fixed', choices=['fixed', 'mice-linear', 'missforest'],
                          help='The way missing values are filled. Either with fixed (mean for metric features, median for ordinal class features),'
                          ' with MICE (Multivariate Imputation by Chained Equations) using a linear bayesian ridge regression model, or with MissForest (MICE and random forest).')
        early_stopping = self.add_argument_group('early stopping', 'Early stopping parameters')
        early_stopping.add_argument('--stopping_patience', type=int, default=40, help='After this number of epochs with loss at least (1 + increase delta) * best loss, the training will stop')
        early_stopping.add_argument('--increase_delta', type=float, default=0.0, help='The early stopping increase delta')

        scheduler = self.add_argument_group('learn rate scheduler', 'Parameters when to reduce the learn rate (on plateau) during training')
        scheduler.add_argument('--patience', type=int, default=10, help='After this number of epochs without drop in loss, the learnrate will be reduced')
        scheduler.add_argument('--threshold', type=float, default=0.0001, help='A drop in loss is considered as (1-threshold) * current best by the scheduler')
        scheduler.add_argument('--factor', type=float, default=0.5, help='The factor by which the learn rate is reduced by the scheduler')
        scheduler.add_argument('--min_lr', type=float, default=0.000001, help='The scheduler will not reduce the learn rate smaller than this parameter')

        loss = self.add_argument_group('loss', 'Parameters for the loss calculation')
        loss.add_argument('--normalization', type=str, default='manners-macro', choices=['manners-macro', 'manners-micro', 'vanilla'],
                            help='The loss normalization mode. With vanilla normalization, loss is calculated for all (also missing) data points. With MANNERS, loss is not calculated for missing data points. '
                            'In macro mode, loss is first averaged per variable and then averaged across variables. In micro mode, loss is averaged over all non-missing data points.')
        loss.add_argument('--missing_weight', type=float, default=1.0, help='The weight of the missing cross entropy loss for the overall reconstruction loss')
        loss.add_argument('--discrete_weight', type=float, default=1.0, help='The weight of the discrete huber loss for the overall reconstruction loss')
        loss.add_argument('--class_weight', type=float, default=1.0, help='The weight of the ordinal classification cross entropy loss for the overall reconstruction loss')
        loss.add_argument('--huber_delta', type=float, default=1.0, help='The delta applied to the discrete huber reconstruction loss')
        loss.add_argument('--gamma', type=float, default=1.0, help='The factor applied to the kl divergence in the overall loss')

        ae = self.add_argument_group('auto encoder', 'Parameters for the autoencoder architecture')
        ae.add_argument('--bottleneck_dim', type=int, default=256, help='The size of the embedding space')
        ae.add_argument('--hidden_dims', type=int, nargs='*', default=[64, 64, 64], help='The number if channels used for each convolutional layer')
        ae.add_argument('--time_kernel_sizes', type=int, nargs='*', default=[4, 3, 5], help='The kernel sizes along the time dimension')
        ae.add_argument('--time_strides', type=int, nargs='*', default=[2, 2, 2], help='The stride of the convolutional kernels along the time dimension')
        ae.add_argument('--time_dilation', type=int, nargs='*', default=[1, 1, 1], help='The dilation of the convolutional kernels along the time dimension')
        ae.add_argument('--dropout', type=float, default=0.0, help='The dropout used between convolutional layers for discriminative models')

    def get_param_dict(self, parsed_args: argparse.Namespace):
        result = {}
        for group in self._action_groups:
            if ' arguments' in group.title:
                continue
            group_result = {}
            for argument in group._group_actions:
                parsed_val = getattr(parsed_args, argument.dest)
                group_result[argument.dest] = parsed_val
            result[group.title] = group_result
        return result
