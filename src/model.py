import torch
from typing import Tuple, List, Union
import numpy as np
from src.config import DISCRETE_FEATURES, CLASS_FEATURES
from src.data_structures import ResultBatch

class ConvBlock(torch.nn.Module):
    """
    A convolutional block that can be configured to use 1D or 2D convolutions, with optional activation, batch normalization, and dropout layers.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        kernel_size (Union[Tuple[int, int], int]): Size of the convolving kernel.
        padding (Union[Tuple[int, int], int]): Zero-padding added to both sides of the input.
        stride (Union[Tuple[int, int], int]): Stride of the convolution.
        dilation (Union[Tuple[int, int], int]): Spacing between kernel elements.
        dropout (float): Dropout probability.
        reverse (bool): If True, uses transposed convolution.
        use_activation (bool, optional): If True, applies GELU activation. Default is True.
        use_batch_norm (bool, optional): If True, applies batch normalization. Default is True.
        use_dropout (bool, optional): If True, applies dropout. Default is True.
        encode_mask (bool, optional): If True, the model will encode the missing mask and use 2D convolution. Default is True.
    """
    def __init__(self, in_dim: int, out_dim: int, kernel_size: Union[Tuple[int, int], int], padding: Union[Tuple[int, int], int],
                 stride: Union[Tuple[int, int], int], dilation: Union[Tuple[int, int], int], dropout: float, reverse: bool,
                 use_activation: bool = True, use_batch_norm: bool = True, use_dropout: bool = True, encode_mask: bool = True):
        super().__init__()
        if encode_mask:
            layer_obj = (torch.nn.ConvTranspose2d if reverse else torch.nn.Conv2d)
        else:
            layer_obj = (torch.nn.ConvTranspose1d if reverse else torch.nn.Conv1d)
        self.main_layer = layer_obj(
            in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, padding_mode=('zeros' if reverse else 'replicate'))
        self.activation = torch.nn.GELU() if use_activation else None
        self.batch_norm = (torch.nn.BatchNorm2d if encode_mask else torch.nn.BatchNorm1d)(num_features=out_dim, affine=True) if use_batch_norm else None
        self.dropout_layer = torch.nn.Dropout(dropout) if use_dropout else None

    def forward(self, x):
        x_out = self.main_layer(x)
        if self.activation is not None:
            x_out = self.activation(x_out)
        if self.batch_norm is not None:
            x_out = self.batch_norm(x_out)
        if self.dropout_layer is not None:
            x_out = self.dropout_layer(x_out)
        return x_out

class FullyConvAutoEncoder(torch.nn.Module):
    """
    A Fully Convolutional Autoencoder with optional variational capabilities.

    Args:
        bottleneck_dim (int): Dimension of the bottleneck layer.
        hidden_dims (List[int]): List of dimensions for hidden layers.
        time_kernel_sizes (List[int]): List of kernel sizes for the time dimension.
        time_dilation (List[int]): List of dilation values for the time dimension.
        time_strides (List[int]): List of stride values for the time dimension.
        dropout (float, optional): Dropout rate. Default is 0.1.
        variational (bool, optional): If True, the autoencoder will be variational. Default is True.
        encode_mask (bool, optional): If True, the model will encode the missing mask and use 2D convolution. Default is True.
    """
    def __init__(self, bottleneck_dim: int, hidden_dims: List[int], time_kernel_sizes: List[int], time_dilation: List[int],
                 time_strides: List[int], dropout: float = 0.0, variational: bool = True, encode_mask: bool = True) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.time_kernel_sizes = time_kernel_sizes
        self.time_strides = time_strides
        self.time_dilation = time_dilation
        self.bottleneck_dim = bottleneck_dim
        self.encode_mask = encode_mask
        # must be at least three layers
        assert len(self.time_kernel_sizes) >= 3
        assert len(self.hidden_dims) == len(self.time_kernel_sizes)
        assert len(self.hidden_dims) == len(self.time_strides)
        assert len(self.hidden_dims) == len(self.time_dilation)
        self.feature_dim = len(DISCRETE_FEATURES) + len(CLASS_FEATURES)
        result_time_dim = 48
        for idx, kernel_size in enumerate(self.time_kernel_sizes):
            result_time_dim = int(np.floor((result_time_dim - self.time_dilation[idx] * (kernel_size -1) - 1)/self.time_strides[idx] + 1))
        if result_time_dim < 1:
            raise AttributeError('Final time dimension must be at least 1, but is ' + str(result_time_dim))
        last_conv_out_dim = self.hidden_dims[-1]
        flattened_dim = last_conv_out_dim * result_time_dim
        if flattened_dim != self.bottleneck_dim:
            raise AttributeError('Time kernel sizes, dilation and strides result in final time dimension of ' + str(result_time_dim) + ', last out dimension of convolution is ' + str(last_conv_out_dim)
                                  + '. Their flattened dimension is then ' + str(flattened_dim) + ' which differs from the specified bottleneck dimension of ' + str(self.bottleneck_dim))
        unflattened_shape = (last_conv_out_dim, 1, result_time_dim) if self.encode_mask else (last_conv_out_dim, result_time_dim)
        self.result_time_dim = result_time_dim
        self.last_conv_out_dim = last_conv_out_dim
        self.dropout = dropout
        if self.encode_mask:
            self.kernel_sizes = [(2 if idx <=2 else 1, time_kernel_size) for idx, time_kernel_size in enumerate(self.time_kernel_sizes)]
            self.dilations = [(1, dilation) for dilation in self.time_dilation]
            self.paddings = [(1 if idx == 0 else 0, 0) for idx in range(len(self.time_kernel_sizes))]
            self.strides = [(1, stride) for stride in self.time_strides]
        else:
            self.kernel_sizes = self.time_kernel_sizes
            self.dilations = self.time_dilation
            self.paddings = [0 for _ in range(len(self.time_kernel_sizes))]
            self.strides = self.time_strides

        self.variational = variational

        shared_encoding_modules = []
        for idx, kernel_size in enumerate(self.kernel_sizes[:-1]):
            in_dim = self.hidden_dims[idx-1] if idx > 0 else self.feature_dim
            out_dim = self.hidden_dims[idx]
            shared_encoding_modules.append(ConvBlock(in_dim=in_dim, out_dim=out_dim, kernel_size=kernel_size, padding=self.paddings[idx], stride=self.strides[idx], dilation=self.dilations[idx], reverse=False,
                                                      use_activation=True, use_batch_norm=True, use_dropout=(not self.variational), dropout=self.dropout, encode_mask=self.encode_mask))

        self.shared_encoder = torch.nn.Sequential(*shared_encoding_modules)

        self.mean_block = torch.nn.Sequential(
            ConvBlock(in_dim=self.hidden_dims[-2], out_dim=self.last_conv_out_dim, kernel_size=self.kernel_sizes[-1], padding=self.paddings[-1], stride=self.strides[-1], dilation=self.dilations[-1],
                      reverse=False, use_activation=False, use_batch_norm=False, use_dropout=False, dropout=self.dropout, encode_mask=self.encode_mask),
            torch.nn.Flatten()
        )

        self.logged_var_block = torch.nn.Sequential(
            ConvBlock(in_dim=self.hidden_dims[-2], out_dim=self.last_conv_out_dim, kernel_size=self.kernel_sizes[-1], padding=self.paddings[-1], stride=self.strides[-1], dilation=self.dilations[-1],
                      reverse=False, use_activation=False, use_batch_norm=False, use_dropout=False, dropout=self.dropout, encode_mask=self.encode_mask),
            torch.nn.Flatten()
        ) if self.variational else None

        decoding_modules = [torch.nn.Unflatten(dim=1, unflattened_size=unflattened_shape)]
        for idx, kernel_size in reversed(list(enumerate(self.kernel_sizes))):
            in_dim = self.hidden_dims[idx] if idx < len(self.hidden_dims) -1 else self.last_conv_out_dim
            out_dim = self.hidden_dims[idx-1] if idx > 0 else self.feature_dim
            decoding_modules.append(ConvBlock(in_dim=in_dim, out_dim=out_dim, kernel_size=kernel_size, padding=self.paddings[idx], stride=self.strides[idx], dilation=self.dilations[idx], reverse=True,
                                              use_activation=(idx > 0), use_batch_norm=(idx > 0), use_dropout=(idx > 0 and not self.variational), dropout=self.dropout, encode_mask=self.encode_mask))


        self.decoder = torch.nn.Sequential(*decoding_modules)
        
        self.coral_biases = torch.nn.ParameterList([
            torch.nn.Parameter(torch.arange(class_info['nof_classes']-1, 0, -1).float() / (class_info['nof_classes']-1))
        for class_info in CLASS_FEATURES.values()])

    def sample(self, z_mean: torch.Tensor, z_logged_var: torch.Tensor):
        if not self.variational:
            raise AttributeError('Sampling only enabled for variational autoencoders')
        std = torch.exp(0.5 * z_logged_var)
        noise = torch.randn_like(std)
        return z_mean + std * noise
    
    def encode(self, x: torch.Tensor):
        # remove stacked mask if not used
        if not self.encode_mask:
            x = x[...,0,:]
        x_encoded = self.shared_encoder(x)
        z_mean = self.mean_block(x_encoded)
        if not self.variational:
            return z_mean
        z_logged_var = self.logged_var_block(x_encoded)
        return z_mean, z_logged_var

    def forward(self, x) -> ResultBatch:
        if not self.variational:
            z = self.encode(x)
            z_mean = None
            z_logged_var = None
        else:
            z_mean, z_logged_var = self.encode(x)
            z = self.sample(z_mean, z_logged_var)
        x_hat = self.decode(z)
        x_hat.z_mean = z_mean
        x_hat.z_logged_var = z_logged_var
        return x_hat
    
    def decode(self, x_btl: torch.Tensor) -> ResultBatch:
        x_out = self.decoder(x_btl)
        nof_disc_features = len(DISCRETE_FEATURES)
        if self.encode_mask:
            x_out_vals = x_out[...,0,:]
            x_out_missing = x_out[...,1,:]
        else:
            x_out_vals = x_out
            x_out_missing = None
        x_disc_out = x_out_vals[:,:nof_disc_features,:]
        x_embedded_classes = x_out_vals[:,nof_disc_features:,:]
        x_classes_transformed = torch.unsqueeze(x_embedded_classes, 3)
        x_class_out = []
        for idx in range(len(CLASS_FEATURES)):
            x_class_out.append(x_classes_transformed[:,idx,...] + self.coral_biases[idx])

        return ResultBatch(missing_logits=x_out_missing, discrete_vals=x_disc_out, ordinal_logits=x_class_out, z=x_btl)