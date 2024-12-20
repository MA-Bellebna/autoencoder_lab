import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import os
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import channel, instance

class Encoder(nn.Module):
    def __init__(self, image_dims, C=220, 
                 activation='relu', channel_norm=True):
        """ 
        Encoder with convolutional architecture proposed in [1].
        Projects image x ([C_in,256,256]) into a feature map of size C x W/16 x H/16
        ========
        Arguments:
        image_dims:  Dimensions of input image, (C_in,H,W)
        batch_size:  Number of instances per minibatch
        C:           Bottleneck depth, controls bits-per-pixel
                     C = {2,4,8,16}

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        
        super(Encoder, self).__init__()
        
        kernel_dim = 3
        filters = (60, 120, 240, 480, 960)

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        im_channels = image_dims[0]
        # assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=0, padding_mode='reflect')
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_downsampling_layers = 4

        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(3)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(1)

        heights = [2**i for i in range(4,9)][::-1]
        widths = heights
        H1, H2, H3, H4, H5 = heights
        W1, W2, W3, W4, W5 = widths 

        # (256,256) -> (256,256), with implicit padding
        self.conv_block1 = nn.Sequential(
            self.pre_pad,
            nn.Conv2d(im_channels, filters[0], kernel_size=(7,7), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
            self.activation(),
        )

        # (256,256) -> (128,128)
        self.conv_block2 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[1], **norm_kwargs),
            self.activation(),
        )

        # (128,128) -> (64,64)
        self.conv_block3 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[2], **norm_kwargs),
            self.activation(),
        )

        # (64,64) -> (32,32)
        self.conv_block4 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[3], **norm_kwargs),
            self.activation(),
        )

        # (32,32) -> (16,16)
        self.conv_block5 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[3], filters[4], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[4], **norm_kwargs),
            self.activation(),
        )
        
        # Project channels onto space w/ dimension C
        # Feature maps have dimension C x W/16 x H/16
        # (16,16) -> (16,16)
        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[4], C, kernel_dim, stride=1),
        )
        
                
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        out = self.conv_block_out(x)
        return out


    def apply_pruning(self, conv_layer, mask):
        """
        Apply the pruning mask to the weights of the Conv2D layer.
        Filters that are pruned (mask == False) will have their weights set to zero.
        The mask will be broadcasted across the input channels and spatial dimensions.
        """
        conv_layer.weight.data *= mask.view(-1, 1, 1, 1).float()  # Broadcasting the mask to match weight shape
        
        # Optional: You can also prune the corresponding bias (if exists)
        if conv_layer.bias is not None:
            conv_layer.bias.data[~mask] = 0  # Set pruned bias to zero


    def prune_filters_by_l1_norm_percentage(self, prune_rate=0.2):
        """
        Prune a percentage of filters based on the L1 norm of the filters' weights.
        The lowest `prune_rate` percentage of filters are pruned.
        
        Arguments:
        - prune_rate: Percentage of filters to prune (between 0 and 1).
        """
        pruned_filters = {}

        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):

                # Compute the L1 norm of each filter
                filter_l1_norm = torch.sum(torch.abs(module.weight.data), dim=(1, 2, 3))

                # Sort the filters by their L1 norm
                sorted_indices = torch.argsort(filter_l1_norm)
                num_filters = filter_l1_norm.size(0)
                num_prune = int(prune_rate * num_filters)  # Number of filters to prune


                if "conv_block_out" in name:
                    print(f"Skipping pruning for {name} to preserve latent space size.")
                    continue
                
                
                # Create mask for keeping and pruning filters
                mask = torch.ones(num_filters, dtype=torch.bool)
                pruned_mask = sorted_indices[:num_prune]  # Indices of filters to prune
                mask[pruned_mask] = False  # Mark the lowest L1 norm filters for pruning

                # Apply pruning mask
                self.apply_pruning(module, mask)

                pruned_filters[name] = {
                    'remaining_filters': mask.sum().item(),
                    'pruned_filters': num_prune,
                    'pruned_indices': pruned_mask
                }
        
        return pruned_filters



    def corr_circle(self, prune_rate, encoder_name):
        """
        Adapted corr_circle to produce a pruning summary matching prune_filters_by_l1_norm_percentage.
        """
        # rate = args.prune_rate

        model_weights = get_model_weights(model)

        model_parameters = np.sum([np.prod(weight.shape) for weight in model_weights])

        print("=" * 50)
        print(f"Baseline Model Parameters: {model_parameters}")
        print("=" * 50)

        pruned_filters = {encoder_name : {}}
        Weights_p = []
        iteration = 0

        while True:
            print(f"Iteration {iteration}")
            total_corr, l2_total = get_cc(model_weights, Weights_p, iteration)

            total_selected = []

            kept_idx = []
            for j, circle_corr in enumerate(total_corr):
                idx = list(range(circle_corr.shape[0]))
                if circle_corr.shape[-1] == 1:
                    kept_idx.append(idx)
                    pruned_filters[encoder_name][f"layer_{i}"] = {
                    "remaining_filters": len(idx),
                    "pruned_filters": 0,
                    "pruned_indices": [],
                    }
                    continue
                    
                cov_mat = cosine_sim(circle_corr, True)
                max_corr = [(i, np.argmax(line)) for i, line in enumerate(cov_mat) if ((np.max(line) > np.cos(10 * np.pi / 180)))]
                selected = mutual_corr(max_corr)
                total_selected.append(selected)
                
                s = []
                for sel in total_selected[j]:
                    if l2_total[j][sel[0]] < l2_total[j][sel[1]]:
                        s.append(sel[0])
                    else:
                        s.append(sel[1])

                kept_idx.append([i for i in idx if i not in s])

            # Ensure consistency with the L1-based method
            for i, indices in enumerate(kept_idx):
                pruned_indices = np.setdiff1d(range(len(model_weights[i])), indices)
                pruned_filters[encoder_name][f"layer_{i}"] = {
                    "remaining_filters": len(indices),
                    "pruned_filters": len(pruned_indices),
                    "pruned_indices": pruned_indices.tolist(),
                }

            print(f"Iteration {iteration} pruning complete.")
            iteration += 1
            break

        return pruned_filters




def get_model_weights(model):
    model_weights = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            model_weights.append(module.weight.data.detach().cpu().numpy())
    return model_weights

def get_cc(model_weights, Weights_p, iteration):
    total_corr = []
    total_magcorr = []
    l2_total = []

    for i, weight in enumerate(tqdm(model_weights)):
    
      weights = weight.reshape(weight.shape[0], -1).transpose()
      if (weights.shape[0] == 1) or (i == (len(model_weights) - 1)):
          l2_total.append(np.zeros(weights.shape[1]))
          total_magcorr.append(np.zeros(weights.shape[1]))
          total_corr.append(np.zeros(weights.shape[1]))
          continue
      
      if iteration == 0:
        pca = PCA().fit(weights)
        weights_p = pca.transform(weights)
        Weights_p.append(weights_p)
      else:
        weights_p = Weights_p[i]
      circle_corr = []
      circ_rad = []
      l2 = []

      for w in weights.transpose():
        corrx = []
        for dim in range(2):
          corrx.append(np.corrcoef(w, weights_p[:, dim])[0, 1])
                   
        circle_corr.append(np.array(corrx))
        corrx = np.array(corrx)
        circ_rad.append(np.sqrt(np.sum(corrx**2)))
          
        l2.append(np.sum(w**2))
        
      total_corr.append(np.array(circle_corr))
      total_magcorr.append(circ_rad)
      l2_total.append(l2)

    return total_corr, l2_total


def mutual_corr(args):
    selected = []
    for e in args:
      if (e[::-1] in args) and (e[::-1] not in selected):
          selected.append(e)
    return selected

def cosine_sim(circle_corr, absolute = True):
    cov_mat = np.zeros((len(circle_corr), len(circle_corr)))
    for i in range(len(circle_corr)):
        for j in range(i, len(circle_corr)):
          if i == j:
              cov_mat[i, j] = 0
              continue
          scalar = np.sum(circle_corr[i] * circle_corr[j])
          norm = np.sqrt(np.sum(circle_corr[i]**2)) * np.sqrt(np.sum(circle_corr[j]**2))
          angle = scalar / norm#(np.arccos(scalar / norm) * 180 / (np.pi))
          cov_mat[i, j] = abs(angle) if absolute else angle  #if angle < 90 else (90 - angle%90)
          cov_mat[j, i] = abs(angle) if absolute else angle  #if angle < 90 else (90 - angle%90)

    cov_mat = cov_mat.astype(np.float32)

    return cov_mat


def create_encoders_with_checkpoints(image_dims, batch_size, bottleneck_depth, checkpoint_paths):
    """
    Function to create encoders for different bitrates and load their respective checkpoints.
    
    Arguments:
    - image_dims: Input image dimensions
    - batch_size: Number of instances per minibatch
    - bottleneck_depth: The depth of the bottleneck for the encoder
    - checkpoint_paths: Dictionary mapping encoder names to checkpoint file paths
    
    Returns:
    - encoders: Dictionary of encoders with loaded checkpoints
    """
    encoders = {}

    for encoder_name, checkpoint_path in checkpoint_paths.items():
        print(f"Creating {encoder_name} and loading checkpoint from {checkpoint_path}...")
        encoder = Encoder(image_dims=image_dims, batch_size=batch_size, C=bottleneck_depth)
        encoder.to(torch.device("cpu"))
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        sd = {}
        for name, value in checkpoint["model_state_dict"].items():
            if "Encoder" in name:
                name = name.split("Encoder.")[1]
                sd[name] = value

    
        # Load the state dictionary into the encoder
        encoder.load_state_dict(sd, strict=False)
        
        encoders[encoder_name] = encoder
    
    return encoders


def analyze_pruned_filters_across_encoders(pruned_results, encoder_name):
    """
    Analyze and compare pruned filters across encoders to identify common unimportant filters.
    Now also granularly breaks it down layer by layer for each block.
    """
    common_pruned_filters = {}

    for layer_name in pruned_results[encoder_name]:
        # Collect the pruned indices from each encoder for the current layer
        pruned_0_45 = set(pruned_results["encoder_0.45_bpp"][layer_name]["pruned_indices"].tolist())
        pruned_0_30 = set(pruned_results["encoder_0.30_bpp"][layer_name]["pruned_indices"].tolist())
        pruned_0_14 = set(pruned_results["encoder_0.14_bpp"][layer_name]["pruned_indices"].tolist())

        # Find common pruned filters across all encoders
        common_pruned = pruned_0_45 & pruned_0_30 & pruned_0_14
            
        pruned_results[encoder_name][layer_name]["pruned_indices"] = torch.tensor(list(common_pruned))

    return pruned_results



def prune_filters_by_l1_norm(encoders, prune_rate=0.2):
    """
    Prune filters using the L1 norm criterion.
    """
    pruned_results = {}
    for encoder_name, encoder in encoders.items():
        print(f"Pruning {encoder_name} using L1 norm...")
        pruned_filters = encoder.prune_filters_by_l1_norm_percentage(prune_rate=prune_rate)
        pruned_results[encoder_name] = pruned_filters
    return pruned_results

def prune_filters_by_cosine_similarity(encoders, similarity_threshold=0.9):
    """
    Prune filters based on cosine similarity between the filter weights.
    Filters with high similarity (above the threshold) across encoders will be pruned.
    """
    pruned_results = {}
    
    def cosine_similarity(tensor1, tensor2):
        dot_product = torch.sum(tensor1 * tensor2)
        norm_tensor1 = torch.norm(tensor1)
        norm_tensor2 = torch.norm(tensor2)
        return dot_product / (norm_tensor1 * norm_tensor2)

    for encoder_name, encoder in encoders.items():
        pruned_filters = {}
        for name, module in encoder.named_modules():
            if isinstance(module, nn.Conv2d):
                if "conv_block_out" in name:
                    print(f"Skipping pruning for {name} to preserve latent space size.")
                weights = module.weight.data
                num_filters = weights.size(0)
                
                # Compute cosine similarity between filters
                similarity_matrix = torch.zeros(num_filters, num_filters)
                for i in range(num_filters):
                    for j in range(i + 1, num_filters):
                        sim = cosine_similarity(weights[i], weights[j])
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
                
                # Prune filters with high cosine similarity
                mask = torch.ones(num_filters, dtype=torch.bool)
                for i in range(num_filters):
                    for j in range(i + 1, num_filters):
                        if similarity_matrix[i, j] > similarity_threshold:
                            mask[j] = False
                
                encoder.apply_pruning(module, mask)
                pruned_filters[name] = {
                    'remaining_filters': mask.sum().item(),
                    'pruned_filters': (~mask).sum().item(),
                    'pruned_indices': (~mask).nonzero().squeeze()
                }
        pruned_results[encoder_name] = pruned_filters
    return pruned_results



def prune_filters_by_weight_distribution(encoders, distribution_threshold=0.2):
    """
    Prune filters based on the similarity of weight distributions.
    Filters with similar distributions will be pruned.
    """
    pruned_results = {}
    
    for encoder_name, encoder in encoders.items():
        pruned_filters = {}
        for name, module in encoder.named_modules():
            if isinstance(module, nn.Conv2d):
                if "conv_block_out" in name:
                    print(f"Skipping pruning for {name} to preserve latent space size.")
                weights = module.weight.data
                num_filters = weights.size(0)
                mask = torch.ones(num_filters, dtype=torch.bool)
                
                # Compare weight distributions (e.g., using histogram distance)
                distributions = []
                for i in range(num_filters):
                    distribution, _ = torch.histogram(weights[i].view(-1), bins=10)
                    distributions.append(distribution)
                
                # Prune filters based on distribution similarity
                for i in range(num_filters):
                    for j in range(i + 1, num_filters):
                        dist = torch.norm(distributions[i] - distributions[j])
                        if dist < distribution_threshold:
                            mask[j] = False
                
                encoder.apply_pruning(module, mask)
                pruned_filters[name] = {
                    'remaining_filters': mask.sum().item(),
                    'pruned_filters': (~mask).sum().item(),
                    'pruned_indices': (~mask).nonzero().squeeze()
                }
        pruned_results[encoder_name] = pruned_filters
    return pruned_results


def analyze_and_display_pruned_filters(encoders):
    """
    Run all pruning criteria and display in-common filters across all encoders.
    Also return the intermediate results for further analysis and visualization.
    """
    prune_rate = 0.4
    similarity_threshold = 0.8
    svd_rank_threshold = 0.9
    distribution_threshold = 0.2

    # L1 Norm Pruning
    print("\n==== L1 Norm Pruning ====")
    pruned_l1 = prune_filters_by_l1_norm(encoders, prune_rate)
    common_l1 = analyze_pruned_filters_across_encoders(pruned_l1)
    print("Common pruned filters (L1 norm):")
    display_granular_common_filters(common_l1)

    # Cosine Similarity Pruning
    print("\n==== Cosine Similarity Pruning ====")
    pruned_similarity = prune_filters_by_cosine_similarity(encoders, similarity_threshold)
    common_similarity = analyze_pruned_filters_across_encoders(pruned_similarity)
    print("Common pruned filters (Cosine Similarity):")
    display_granular_common_filters(common_similarity)

    # Weight Distribution Pruning
    print("\n==== Weight Distribution Pruning ====")
    pruned_distribution = prune_filters_by_weight_distribution(encoders, distribution_threshold)
    common_distribution = analyze_pruned_filters_across_encoders(pruned_distribution)
    print("Common pruned filters (Weight Distribution):")
    display_granular_common_filters(common_distribution)

    # Return the results for further analysis or visualization
    return {
        'L1_Norm': {
            'pruned': pruned_l1,
            'common': common_l1
        },
        'Cosine_Similarity': {
            'pruned': pruned_similarity,
            'common': common_similarity
        },
        'Weight_Distribution': {
            'pruned': pruned_distribution,
            'common': common_distribution
        }
    }

def display_granular_common_filters(common_filters):
    """
    Displays the common filters for each block and layer, breaking it down granularly.
    """
    current_block = ""
    
    for layer_name, common in common_filters.items():
        block_name = layer_name.split('.')[0]  # Extract block (e.g., 'conv_block1')
        
        # Detect block changes and print the block header
        if block_name != current_block:
            print(f"\n--- {block_name} ---")
            current_block = block_name
        
        print(f"Layer {layer_name}: Common filters: {common}")


    
# Function to visualize and analyze the results
def visualize_analysis(weight_distributions, similarities, svd_pruning):
    print("\nWeight Distributions Comparison:")
    for encoder, distribution in weight_distributions.items():
        print(f"{encoder}: {distribution}")
    
    print("\nCosine Similarity Between Encoders (compared to encoder_0.45_bpp):")
    for encoder, similarity in similarities.items():
        print(f"{encoder}: {similarity}")
    
    print("\nSVD-Based Pruning Results (Number of retained filters):")
    for encoder, svd_result in svd_pruning.items():
        print(f"{encoder}: {svd_result}")


def display_encoder_architecture(encoder):
    """
    Display the architecture of the encoder in a readable format.
    """
    print("Encoder Architecture Details:\n")
    for name, module in encoder.named_modules():
        print(f"Layer: {name}, Type: {module.__class__.__name__}")
 

# Function to print the number of parameters in the model
def count_parameters(model):
    total_params = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            total_params += layer_params
    
    return total_params


def compute_conv_flops(input_shape, output_shape, layer):
    """
    Compute the FLOPs for a given Conv2D layer.
    
    Arguments:
    - input_shape: Tuple (Batch size, Input channels, Input height, Input width).
    - output_shape: Tuple (Batch size, Output channels, Output height, Output width).
    - layer: The Conv2D layer.
    
    Returns:
    - flops: The number of FLOPs for this layer.
    """
    batch_size, input_channels, input_height, input_width = input_shape
    batch_size, output_channels, output_height, output_width = output_shape
    
    kernel_size = layer.kernel_size
    flops_per_instance = (
        2 * output_channels * output_height * output_width *
        kernel_size[0] * kernel_size[1] * input_channels
    )
    
    total_flops = batch_size * flops_per_instance
    return total_flops

def register_hooks(model):
    """
    Registers hooks on each Conv2D layer to capture input and output shapes.
    Also calculates and sums up FLOPs for each layer.
    
    Arguments:
    - model: The PyTorch model.
    
    Returns:
    - total_flops: The total number of FLOPs for the model.
    """
    conv_flops = {}

    def hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            input_shape = input[0].shape  # Shape of input tensor (Batch, Channels, Height, Width)
            output_shape = output.shape  # Shape of output tensor
            flops = compute_conv_flops(input_shape, output_shape, module)
            conv_flops[module] = flops

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook))
    
    return hooks, conv_flops




# Counting Parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Compute FLOPs for Conv2D Layer
def compute_conv_flops(input_shape, output_shape, layer):
    batch_size, input_channels, input_height, input_width = input_shape
    batch_size, output_channels, output_height, output_width = output_shape
    
    kernel_size = layer.kernel_size
    flops_per_instance = (
        2 * output_channels * output_height * output_width *
        kernel_size[0] * kernel_size[1] * input_channels
    )
    
    total_flops = batch_size * flops_per_instance
    return total_flops


def print_flops(conv_flops):
    total_flops = 0
    for layer, flops in conv_flops.items():
        # print(f"Layer {layer}: {flops / 1e6:.4f} MFLOPs")
        total_flops += flops
    print(f"Total FLOPs: {total_flops / 1e6:.4f} MFLOPs")

    return total_flops

import copy

def apply_physical_pruning(original_encoder, pruned_filters_info):
    """
    Physically prune the filters in the encoder based on the pruned filters info.
    Replace the convolutional layers with layers having fewer filters (pruned).
    Also cascade pruning to subsequent layers (input channels of the next layer).
    
    Arguments:
    - original_encoder: The encoder model to prune.
    - pruned_filters_info: A dictionary containing pruned filters information for each layer.
    """
    # Clone the original encoder so we don't modify it
    pruned_encoder = copy.deepcopy(original_encoder)
    previous_pruned_filters = None  # Track pruned output channels from the previous layer

    for layer_name, pruned_info in pruned_filters_info.items():
        # Access the Conv2d layer by traversing the model hierarchy
        modules = layer_name.split('.')
        conv_layer = pruned_encoder
        for mod in modules:
            conv_layer = getattr(conv_layer, mod)

        # Get the pruned indices (those we need to remove)
        total_filters = conv_layer.weight.size(0)
        pruned_indices = pruned_info['pruned_indices'].cpu().numpy()        
        remaining_filter_indices = np.setdiff1d(np.arange(total_filters), pruned_indices)

        print(f"Layer {layer_name}: Pruned Filters = {len(pruned_indices)}, Remaining Filters = {len(remaining_filter_indices)}")
        
        if isinstance(conv_layer, nn.Conv2d):
            # Prune the Conv2d layer weights
            original_weights = conv_layer.weight.data.cpu().numpy()
            original_bias = conv_layer.bias.data.cpu().numpy() if conv_layer.bias is not None else None

            new_out_channels = len(remaining_filter_indices)
            new_weights = original_weights[remaining_filter_indices, :, :, :]
            new_bias = original_bias[remaining_filter_indices] if original_bias is not None else None

            # Cascade pruning to the input channels of the next layer (from previous pruned filters)
            if previous_pruned_filters is not None:
                new_weights = new_weights[:, previous_pruned_filters, :, :]
            
            # Recreate Conv2d layer with pruned weights
            new_layer = nn.Conv2d(
                in_channels=conv_layer.in_channels - len(previous_pruned_filters) if previous_pruned_filters is not None else conv_layer.in_channels,
                out_channels=new_out_channels,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.padding,
                dilation=conv_layer.dilation,
                groups=conv_layer.groups,
                bias=(new_bias is not None),
                padding_mode=conv_layer.padding_mode
            )
            new_layer.weight.data = torch.from_numpy(new_weights).to(conv_layer.weight.device)
            if new_bias is not None:
                new_layer.bias.data = torch.from_numpy(new_bias).to(conv_layer.bias.device)

            # Access the Sequential block containing both Conv2d and ChannelNorm2D
            parent_module = pruned_encoder
            for mod in modules[:-1]:  # Traverse only up to the Sequential block level
                parent_module = getattr(parent_module, mod)
            
            # Replace Conv2d layer in the Sequential block
            setattr(parent_module, modules[-1], new_layer)

            # Now, access and prune the normalization layer within the same Sequential block

            norm_layer = parent_module[2]  # ChannelNorm2D is the second layer in each Sequential block
            
            if isinstance(norm_layer, ChannelNorm2D):
                # Prune normalization parameters to match the pruned Conv2d layer
                if hasattr(norm_layer, 'beta'):
                    norm_layer.beta = nn.Parameter(norm_layer.beta.data[:, remaining_filter_indices])
                if hasattr(norm_layer, 'gamma'):
                    norm_layer.gamma = nn.Parameter(norm_layer.gamma.data[:, remaining_filter_indices])
                
                print(f"Adjusted normalization layer in {layer_name} to match pruned filters.")

            print(f"Layer {layer_name}: Original Filters = {total_filters}, Pruned Filters = {np.size(pruned_indices)}")
            previous_pruned_filters = remaining_filter_indices  # Track remaining filters for the next layer

    # Handle conv_block_out separately
    conv_block_out_layer = pruned_encoder.conv_block_out[1]  # Access the Conv2d layer in conv_block_out
    if isinstance(conv_block_out_layer, nn.Conv2d) and previous_pruned_filters is not None:
        original_weights = conv_block_out_layer.weight.data.cpu().numpy()
        original_bias = conv_block_out_layer.bias.data.cpu().numpy() if conv_block_out_layer.bias is not None else None
        
        # Adjust input channels to match pruned filters from conv_block5, but keep output channels intact
        new_weights = original_weights[:, previous_pruned_filters, :, :]

        # Create a new Conv2d layer with adjusted input channels but the same output channels
        final_layer = nn.Conv2d(
            in_channels=len(previous_pruned_filters),
            out_channels=conv_block_out_layer.out_channels,  # Preserve original output channels
            kernel_size=conv_block_out_layer.kernel_size,
            stride=conv_block_out_layer.stride,
            padding=conv_block_out_layer.padding,
            dilation=conv_block_out_layer.dilation,
            groups=conv_block_out_layer.groups,
            bias=(original_bias is not None),
            padding_mode=conv_block_out_layer.padding_mode
        )
        
        # Assign pruned weights and bias
        final_layer.weight.data = torch.from_numpy(new_weights).to(conv_block_out_layer.weight.device)
        if original_bias is not None:
            final_layer.bias.data = torch.from_numpy(original_bias).to(conv_block_out_layer.bias.device)
        
        # Replace the layer in conv_block_out
        pruned_encoder.conv_block_out[1] = final_layer
        print(f"Adjusted input channels of conv_block_out to match pruned filters from conv_block5.")
    
    return pruned_encoder

# Example function to apply pruning to a specific encoder and compare before and after
def prune_and_compare(encoder_name, encoders, pruned_results, input_size, orig_flops, orig_param):

    original_encoder = encoders[encoder_name]
    pruned_filters_info = pruned_results[encoder_name]
    
    # Print original number of parameters and FLOPs
    # original_params = count_parameters(original_encoder)
    hooks, conv_flops = register_hooks(original_encoder)
    print_flops(conv_flops)
    output = original_encoder(torch.randn(input_size))
    # orig_flops = print_flops(conv_flops, logger)
    
    print(f"\nOriginal Encoder '{encoder_name}'")
    print(f"Number of Parameters: {orig_param}")

    # Apply physical pruning to the encoder copy
    print("\nApplying physical pruning...")
    pruned_encoder = apply_physical_pruning(original_encoder, pruned_filters_info)

    # Print pruned number of parameters and FLOPs
    pruned_params = count_parameters(pruned_encoder)
    hooks, pruned_flops = register_hooks(pruned_encoder)

    output = pruned_encoder(torch.randn(input_size))
    pruned_flops = print_flops(pruned_flops)

    print(f"\nPruned Encoder '{encoder_name}'")
    print(f"Number of Parameters: {pruned_params}")

    param_ratio = 1 - pruned_params / orig_param
    flops_ratio = 1 - pruned_flops / orig_flops

    print("Pruning param ratio is : {:.3f}".format(param_ratio))
    print("Pruning flops ratio is : {:.3f}".format(flops_ratio))
    # Print encoder details for original and pruned versions

    print_encoder_pruning_details(original_encoder, pruned_encoder, logger)

    return original_encoder, pruned_encoder, flops_ratio, param_ratio


# Example to extract in-common filters, apply pruning to one encoder, and compare
def extract_and_prune_encoder(encoders, encoder_name="encoder_0.45_bpp", input_size=(1, 3, 256, 256)):
    # Use L1 Norm as the pruning criteria
    pruned_l1 = prune_filters_by_l1_norm(encoders, prune_rate=0.2)

    # Analyze pruned filters across encoders to find common ones
    common_l1 = analyze_pruned_filters_across_encoders(pruned_l1, encoder_name)

    # Physically prune the specific encoder (e.g., "encoder_0.45_bpp")
    original_encoder, pruned_encoder = prune_and_compare(encoder_name, encoders, common_l1, input_size)

    verify_pruned_weights(original_encoder, pruned_encoder, common_l1[encoder_name])

    return original_encoder, pruned_encoder


def print_encoder_pruning_details(original_encoder, pruned_encoder):
    """
    Print the details of each encoder side by side.
    Show the size of filters (output channels) within each layer (original and pruned).
    
    Arguments:
    - original_encoder: The original encoder model.
    - pruned_encoder: The pruned encoder model.
    """
    print("{:<25} {:<25} {:<25}".format("Layer", "Original Filters", "Pruned Filters"))
    print("="*75)

    for (layer_name, layer), (_, pruned_layer) in zip(original_encoder.named_modules(), pruned_encoder.named_modules()):
        if isinstance(layer, nn.Conv2d):
            # Get original and pruned number of filters (output channels)
            original_filters = layer.weight.size(0)
            pruned_filters = pruned_layer.weight.size(0)

            # Print side by side comparison
            print("{:<25} {:<25} {:<25}".format(
                layer_name,
                original_filters,
                pruned_filters
            ))

    print("\nPruning details printed for all encoders.")





def verify_pruned_weights(original_encoder, pruned_encoder, pruned_filters_info):
    """
    Verifies that the pruned encoder only contains the remaining weights as specified
    in the pruned_filters_info. This includes checking both filters and input channels.

    Arguments:
    - original_encoder: The original encoder model.
    - pruned_encoder: The pruned encoder model.
    - pruned_filters_info: A dictionary containing pruned filters information for each layer.
    - logger: Logger instance for logging messages.
    """
    print("Verifying pruned weights and channels...")

    previous_remaining_indices = None  # Track remaining output channels from the previous layer

    for layer_name, pruned_info in pruned_filters_info.items():
        print(f"Verifying layer: {layer_name}")

        # Retrieve the original and pruned layers
        modules = layer_name.split('.')
        original_layer = original_encoder
        pruned_layer = pruned_encoder

        # Access the original and pruned Conv2D layers
        try:
            for mod in modules:
                original_layer = getattr(original_layer, mod)
                pruned_layer = getattr(pruned_layer, mod)
        except AttributeError:
            print(f"Layer {layer_name} not found in model. Please check layer names.")
            continue

        # Get remaining indices for the current layer
        pruned_indices = pruned_info['pruned_indices'].cpu().numpy()
        remaining_indices = np.setdiff1d(np.arange(original_layer.weight.size(0)), pruned_indices)

        # Filter weights to match the pruned layer's dimensions
        original_weights = original_layer.weight.data[remaining_indices].cpu()
        pruned_weights = pruned_layer.weight.data.cpu()

        
        print(original_weights.shape)
        print(pruned_weights.shape)
        # Ensure output channel shapes match
        if original_weights.shape != pruned_weights.shape or not torch.allclose(original_weights, pruned_weights, atol=1e-6):
            print(f"Weight mismatch detected in layer {layer_name}!")
            return False

        if original_layer.bias is not None:
            original_bias = original_layer.bias.data[remaining_indices].cpu()
            pruned_bias = pruned_layer.bias.data.cpu()
            if original_bias.shape != pruned_bias.shape or not torch.allclose(original_bias, pruned_bias, atol=1e-6):
                print(f"Bias mismatch detected in layer {layer_name}!")
                return False

        # Channel pruning verification: Check that the pruned layer's input channels match previous layer's output
        if previous_remaining_indices is not None:
            # Verify that the pruned layer's input channels match previous layer's output
            pruned_input_channels = pruned_layer.weight.data[:, previous_remaining_indices, :, :].cpu()
            original_input_channels = original_layer.weight.data[remaining_indices][:, previous_remaining_indices, :, :].cpu()

            # Ensure input channel shapes match
            if original_input_channels.shape != pruned_input_channels.shape or not torch.allclose(original_input_channels, pruned_input_channels, atol=1e-6):
                print(f"Channel mismatch detected in input channels of layer {layer_name}!")
                return False

        print(f"Layer {layer_name} passed verification.")

        # Update previous remaining indices for channel verification in the next layer
        previous_remaining_indices = remaining_indices

    # Special handling for the final layer
    if hasattr(pruned_encoder, 'conv_block_out'):
        final_layer = pruned_encoder.conv_block_out[1]
        original_final_layer = original_encoder.conv_block_out[1]
        if previous_remaining_indices is not None:
            final_layer_weights = final_layer.weight.data[:, previous_remaining_indices, :, :].cpu()
            original_final_weights = original_final_layer.weight.data[:, previous_remaining_indices, :, :].cpu()

            if original_final_weights.shape != final_layer_weights.shape or not torch.allclose(original_final_weights, final_layer_weights, atol=1e-6):
                print("Mismatch detected in final layer input channels after pruning.")
                return False

    print("All pruned layers and channels have been verified successfully.")
    return True



def get_codebook(stage_pruning_ratio, depth, skip_layers, Prune_ResLayers = False):
    
    n = int((depth - 2) / 6)
    filter_size = [16, 32, 64]
    interval = np.arange(3)
    print("pruned_filter_size", np.round(np.array(filter_size) * np.array(stage_pruning_ratio)))
    pruned_filter_size = np.zeros(len(filter_size)) - 1
    for stage in interval: 
        if stage_pruning_ratio[stage] == 0:
            continue
        # pruned_filter_size = np.zeros(len(filter_size)) - 1
        pruned_filter_size[stage] = np.round(filter_size[stage] * (1 - stage_pruning_ratio[stage]))
        # if stage == 0:
        #     ratio = 1 - (pruning_ratio * 1 / filter_size[stage])
        # elif stage == 1:
        #     ratio = 1 - (pruning_ratio * 2 / filter_size[stage])
        # else:
        #     ratio = 1 - (pruning_ratio * 3 / filter_size[stage])

        rem_filters = []
        residual_idx = []

        rem_filters.append(pruned_filter_size[0] if Prune_ResLayers else -1)
        for s in range(3):
            for b in range(n):
                if s > 0 and b == 0:
                    rem_filters.append(pruned_filter_size[s])
            
                    rem_filters.append(pruned_filter_size[s] if Prune_ResLayers else -1)
                    rem_filters.append(pruned_filter_size[s] if Prune_ResLayers else -1)
                    residual_idx.append((s * n + b) * 2 + s - 1)
                    residual_idx.append((s * n + b) * 2 + s + 1)
                    continue

                rem_filters.append(pruned_filter_size[s])
                rem_filters.append(pruned_filter_size[s] if Prune_ResLayers else -1)
                residual_idx.append((s * n + b) * 2 + s)

        residual_idx.append(depth)
        rem_filters = np.array(rem_filters)
        residual_idx = np.array(residual_idx)
        rem_filters[np.argwhere(rem_filters == -2).squeeze()] == -1
        if skip_layers is not None:
            rem_filters[skip_layers] = -1
    
        pruning_state = np.zeros(len(rem_filters))
        pruning_state[np.argwhere(np.array(rem_filters) == -1)] = 1

    print("pruning_state", pruning_state)
    return pruning_state, residual_idx

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == "__main__":
    B = 1  # Batch size
    C = 220  # Bottleneck depth
    x = torch.randn((B, 3, 256, 256))  # Random input tensor (batch of 2 images)
    input_size = (B, 3, 256, 256)

    checkpoint_paths = {
        "encoder_0.45_bpp": "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_ln_adapt_linear_origW_2024_08_13_12_16/HiFiC_Zoom_FFX/best_checkpoint.pt",
        # "encoder_0.30_bpp": "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_med_adapt_linear_origW_2024_09_11_22_41/HiFiC_Zoom_FFX/best_checkpoint.pt",
        # "encoder_0.14_bpp": "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_low_adapt_linear_origW_2024_09_11_22_41/HiFiC_Zoom_FFX/best_checkpoint.pt"
    }


    # Create encoder instances and load checkpoints
    encoders = create_encoders_with_checkpoints(image_dims=(3, 256, 256), batch_size=B, bottleneck_depth=C, checkpoint_paths=checkpoint_paths)

    # # Extract common filters and prune one encoder
    # orig_model, pruned_model = extract_and_prune_encoder(encoders, encoder_name="encoder_0.45_bpp", input_size=input_size)

    # Instantiate the model and move to appropriate device

    encoder_name = "encoder_0.45_bpp"
    model = encoders[encoder_name]
    model.eval()  # Ensure evaluation mode for testing


    # selected_filter = [np.arange(weight.shape[0]) for weight in get_model_weights(model)]

    prune_rate = 0.5
    corr_pruned_filters = model.corr_circle(prune_rate, encoder_name)

    # Physically prune the specific encoder (e.g., "encoder_0.45_bpp")
    original_encoder, pruned_encoder = prune_and_compare(encoder_name, encoders, corr_pruned_filters, input_size)

    verify_pruned_weights(original_encoder, pruned_encoder, corr_pruned_filters[encoder_name])