import torch.nn as nn
import numpy as np
import torch
import math


class Mapping(nn.Module):
    """
    A mapping module that transforms input feature maps through a convolutional layer followed by batch normalization.
    
    Attributes:
        preconv (nn.Conv2d): A 2D convolutional layer that reduces the number of input channels to the specified output dimension.
        preconv_bn (nn.BatchNorm2d): Batch normalization applied to the output of the convolutional layer.
    """

    def __init__(self, in_dimension, out_dimension):
        """
        Initializes the Mapping module with the specified input and output dimensions.

        Args:
            in_dimension (int): The number of input channels.
            out_dimension (int): The number of output channels.
        """
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)  # 1x1 convolution
        self.preconv_bn = nn.BatchNorm2d(out_dimension)  # Batch normalization layer

    def forward(self, x):
        """
        Forward pass through the Mapping module.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size, 
                              C is the number of input channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: Output tensor after applying convolution and batch normalization.
        """
        x = self.preconv(x)  # Apply convolution
        x = self.preconv_bn(x)  # Apply batch normalization
        return x  # Return the transformed output

def conv3x3x3(in_channel, out_channel):
    """
    Creates a 3D convolutional layer followed by batch normalization.

    Args:
        in_channel (int): The number of input channels.
        out_channel (int): The number of output channels.

    Returns:
        nn.Sequential: A sequential model containing the 3D convolutional layer and batch normalization.
    """
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),  # 3x3x3 convolution
        nn.BatchNorm3d(out_channel),  # Batch normalization for 3D convolution
        # nn.ReLU(inplace=True)  # Uncomment to add ReLU activation (optional)
    )
    return layer  # Return the sequential model


import torch
import torch.nn as nn
import torch.nn.functional as F

class residual_block(nn.Module):
    """
    A residual block that implements the residual learning framework to facilitate training of deep neural networks.

    This block consists of three convolutional layers with batch normalization and ReLU activation. The output of the block 
    is computed by adding the input tensor to the output of the third convolutional layer (skip connection).

    Attributes:
        conv1 (nn.Sequential): The first convolutional layer.
        conv2 (nn.Sequential): The second convolutional layer.
        conv3 (nn.Sequential): The third convolutional layer.
    """

    def __init__(self, in_channel, out_channel):
        """
        Initializes the residual block with the specified input and output channels.

        Args:
            in_channel (int): The number of input channels for the first convolutional layer.
            out_channel (int): The number of output channels for the convolutional layers.
        """
        super(residual_block, self).__init__()
        
        # Initialize the convolutional layers
        self.conv1 = conv3x3x3(in_channel, out_channel)  # First convolution layer
        self.conv2 = conv3x3x3(out_channel, out_channel)  # Second convolution layer
        self.conv3 = conv3x3x3(out_channel, out_channel)  # Third convolution layer

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W), where N is the batch size, 
                              C is the number of input channels, D is the depth, H is the height, and W is the width.

        Returns:
            torch.Tensor: Output tensor after applying the convolutional layers and adding the input (skip connection).
        """
        x1 = F.relu(self.conv1(x), inplace=True)  # Apply first conv layer and ReLU activation
        x2 = F.relu(self.conv2(x1), inplace=True)  # Apply second conv layer and ReLU activation
        x3 = self.conv3(x2)  # Apply third conv layer

        # Add the input to the output of the third conv layer (skip connection) and apply ReLU
        out = F.relu(x1 + x3, inplace=True)  
        return out  # Return the output tensor


import torch
import torch.nn as nn

class D_Res_3d_CNN(nn.Module):
    """
    A 3D Residual Convolutional Neural Network (CNN) that implements a series of residual blocks 
    followed by max pooling and a final convolutional layer. This architecture is designed to extract
    features from 3D input data.

    Attributes:
        block1 (residual_block): The first residual block.
        maxpool1 (nn.MaxPool3d): The first max pooling layer.
        block2 (residual_block): The second residual block.
        maxpool2 (nn.MaxPool3d): The second max pooling layer.
        conv (nn.Conv3d): The final convolutional layer that reduces the output to 32 channels.
    """

    def __init__(self, in_channel, out_channel1, out_channel2):
        """
        Initializes the 3D Residual CNN with the specified input and output channels.

        Args:
            in_channel (int): The number of input channels for the first residual block.
            out_channel1 (int): The number of output channels for the first residual block.
            out_channel2 (int): The number of output channels for the second residual block.
        """
        super(D_Res_3d_CNN, self).__init__()
        
        # Initialize the first residual block and pooling layer
        self.block1 = residual_block(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        
        # Initialize the second residual block and pooling layer
        self.block2 = residual_block(out_channel1, out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        
        # Final convolutional layer to reduce output channels
        self.conv = nn.Conv3d(in_channels=out_channel2, out_channels=32, kernel_size=3, bias=False)

    def forward(self, x):
        """
        Forward pass through the 3D Residual CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W) where N is the batch size, 
                              C is the number of channels, D is the depth, H is the height, and W is the width.

        Returns:
            torch.Tensor: Output tensor after processing through the residual blocks, pooling layers, and final convolution.
                          The output shape will be (N, 160).
        """
        # Add an additional channel dimension to the input tensor
        x = x.unsqueeze(1)  # Shape: (N, 1, 100, 9, 9)

        # Process through the first residual block and pooling layer
        x = self.block1(x)  # Shape after block1: (N, out_channel1, 100, 9, 9)
        x = self.maxpool1(x)  # Shape after maxpool1: (N, out_channel1, 25, 5, 5)

        # Process through the second residual block and pooling layer
        x = self.block2(x)  # Shape after block2: (N, out_channel2, 25, 5, 5)
        x = self.maxpool2(x)  # Shape after maxpool2: (N, out_channel2, 7, 3, 3)

        # Apply the final convolutional layer
        x = self.conv(x)  # Shape after conv: (N, 32, 5, 1, 1)

        # Flatten the output tensor
        x = x.view(x.shape[0], -1)  # Shape: (N, 160)

        # Return the flattened output tensor
        return x  # Returns the feature vector for classification or further processing



#############################################################################################################

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    """
    Calculate a coefficient that varies over iterations.

    The coefficient is computed using a logistic function that scales between `low` and `high`.
    This is often used in training to modulate the contribution of certain components of the loss.

    Args:
        iter_num (int): The current iteration number.
        high (float): The upper bound of the coefficient.
        low (float): The lower bound of the coefficient.
        alpha (float): The steepness of the logistic function.
        max_iter (float): The maximum number of iterations to normalize the coefficient.

    Returns:
        float: A calculated coefficient in the range [low, high].
    """
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    """
    Create a gradient reversal hook for the given coefficient.

    This function defines a hook that will multiply the gradient by the negative coefficient,
    effectively reversing the direction of the gradient during backpropagation.

    Args:
        coeff (float): The coefficient to multiply the gradient by.

    Returns:
        function: A function that takes the gradient as input and returns the modified gradient.
    """
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

class DomainClassifier(nn.Module):
    """
    A domain classifier for distinguishing between different domains.

    This model consists of multiple fully connected layers with ReLU activations and dropout for regularization.
    It outputs a single value representing the predicted domain.

    Attributes:
        layer (nn.Sequential): A sequential container of fully connected layers.
        domain (nn.Linear): The final layer that outputs the domain prediction.
    """
    def __init__(self):
        """
        Initializes the DomainClassifier with fully connected layers.
        """
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024),  # First fully connected layer
            nn.ReLU(),              # ReLU activation
            nn.Dropout(0.5),        # Dropout for regularization

            nn.Linear(1024, 1024),  # Second fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),  # Third fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),  # Fourth fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.domain = nn.Linear(1024, 1)  # Output layer for domain classification

    def forward(self, x, iter_num):
        """
        Forward pass through the domain classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 1024) where N is the batch size.
            iter_num (int): The current iteration number for coefficient calculation.

        Returns:
            torch.Tensor: The output tensor representing the predicted domain.
        """
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10, 10000.0)  # Calculate the coefficient
        x.register_hook(grl_hook(coeff))  # Register the gradient reversal hook
        x = self.layer(x)  # Pass through the fully connected layers
        domain_y = self.domain(x)  # Get the domain prediction
        return domain_y

class RandomLayer(nn.Module):
    """
    A layer that applies a random linear transformation to its input.

    This layer generates a random weight matrix for each input dimension and applies it to the input data.
    The outputs are scaled by the output dimension.

    Attributes:
        input_num (int): The number of input dimensions.
        output_dim (int): The number of output dimensions.
        random_matrix (list): A list of randomly generated weight matrices for each input dimension.
    """
    def __init__(self, input_dim_list=[], output_dim=1024):
        """
        Initializes the RandomLayer with the specified input dimensions and output dimension.

        Args:
            input_dim_list (list): A list containing the number of input dimensions for each input tensor.
            output_dim (int): The number of output dimensions after transformation.
        """
        super(RandomLayer, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_num = len(input_dim_list)  # Number of input tensors
        self.output_dim = output_dim  # Set output dimension
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim).to(device) for i in range(self.input_num)]  # Create random matrices

    def forward(self, input_list):
        """
        Forward pass through the random layer.

        Args:
            input_list (list): A list of input tensors, each corresponding to an entry in input_dim_list.

        Returns:
            torch.Tensor: The output tensor after applying the random transformations.
        """
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]  # Apply random matrices
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))  # Normalize the first output
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)  # Element-wise multiplication of all outputs
        return return_tensor

    def cuda(self):
        """
        Move the random layer to the GPU.

        Overrides the default cuda() method to move the random matrices to the GPU.
        """
        super(RandomLayer, self).cuda()  # Move to GPU
        self.random_matrix = [val.cuda() for val in self.random_matrix]  # Move random matrices to GPU


class Network(nn.Module):
    """
    Defines the main network for few-shot learning and domain adaptation in hyperspectral image classification.

    Attributes:
    feature_encoder (D_Res_3d_CNN): The feature extraction module using 3D CNN with residual blocks.
    final_feat_dim (int): The final feature dimension.
    classifier (nn.Linear): The linear classifier.
    target_mapping (Mapping): Mapping layer for target domain data.
    source_mapping (Mapping): Mapping layer for source domain data.
    """
    def __init__(self, FEATURE_DIM, SRC_INPUT_DIMENSION, TAR_INPUT_DIMENSION, N_DIMENSION, CLASS_NUM):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1, 8, 16)
        self.final_feat_dim = FEATURE_DIM  # 64+32
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, x, domain='source'):
        """
        Forward pass through the network.

        Parameters:
        x (torch.Tensor): Input tensor.
        domain (str): The domain of the input data ('source' or 'target').

        Returns:
        feature (torch.Tensor): Extracted feature tensor.
        output (torch.Tensor): Classifier output tensor.
        """
        if domain == 'target':
            x = self.target_mapping(x)
        elif domain == 'source':
            x = self.source_mapping(x)

        feature = self.feature_encoder(x)
        output = self.classifier(feature)
        return feature, output
