import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def save_BLD_as_image(tensor, index, filename):
    # Input tensor size: [B, L, D]
    # Check if the index is within the valid range
    if index < 0 or index >= tensor.size(0):
        raise ValueError("Index out of range")
    
    # Extract the [L, D] tensor
    selected_tensor = tensor[index]

    # Convert the tensor to a numpy array
    selected_array = selected_tensor.detach().cpu().numpy().T

    # Plot the array as an image
    plt.imshow(selected_array, cmap='gray')
    plt.axis('off')
    
    # Save the image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def save_summed_BLD_graph(tensor, index, filename):
    # Check if the index is within the valid range
    if index < 0 or index >= tensor.size(0):
        raise ValueError("Index out of range")
    
    # Extract the [L, D] tensor
    selected_tensor = tensor[index]

    # Sum along the D dimension to get a [L] tensor
    # summed_tensor = selected_tensor.sum(dim=1)
    summed_tensor = selected_tensor[:,100]

    # Convert the tensor to a numpy array
    summed_array = summed_tensor.detach().cpu().numpy()

    # Plot the summed values as a graph
    plt.plot(summed_array)
    plt.xlabel('L Dimension')
    plt.ylabel('Summed Value')
    plt.title('Summed Values along D Dimension')
    
    # Save the graph
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def save_BLD_zero_ratio_graph(tensor, index, filename):
    # Check if the index is within the valid range
    if index < 0 or index >= tensor.size(0):
        raise ValueError("Index out of range")
    
    # Extract the [L, D] tensor
    selected_tensor = tensor[index]

    # Calculate the ratio of zeros along the D dimension
    zero_ratio = (selected_tensor == 0).float().mean(dim=1)

    # Convert the tensor to a numpy array
    zero_ratio_array = zero_ratio.detach().cpu().numpy()

    # Plot the zero ratio values as a graph
    plt.plot(zero_ratio_array)
    plt.xlabel('L Dimension')
    plt.ylabel('Zero Ratio')
    plt.title('Zero Ratio along D Dimension')
    
    # Save the graph
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def save_BLD_path_as_image(tensor, index, filename, path='pathfinder'):
    # Check if the index is within the valid range
    if index < 0 or index >= tensor.size(0):
        raise ValueError("Index out of range")
    
    # Extract the [L, D] tensor
    selected_tensor = tensor[index,:,127]
    # selected_tensor = tensor[index].sum(dim=-1)
    
    # selected_tensor = tensor[index,0,:]

    # Convert the tensor to a numpy array
    if path == 'pathfinder':
        selected_array = selected_tensor.detach().cpu().view(32,32).numpy()
        # selected_array = selected_tensor.detach().cpu().view(16,16).numpy()
    elif path == 'pathx':
        selected_array = selected_tensor.detach().cpu().view(128,128).numpy()

    # Plot the array as an image
    plt.imshow(selected_array, cmap='gray')
    plt.axis('off')
    
    # Save the image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def save_BLD_path_spk_frq_as_image(tensor, index, filename, path='pathfinder'):
    # Check if the index is within the valid range
    if index < 0 or index >= tensor.size(0):
        raise ValueError("Index out of range")
    
    # Extract the [L, D] tensor
    selected_tensor = tensor[index]

    # Sum along the D dimension to get a [L] tensor
    pix_frq_img = selected_tensor.sum(dim=-1) / selected_tensor.shape[-1]

    # Convert the tensor to a numpy array
    pix_frq_img = pix_frq_img.detach().cpu().view(32,32).numpy()

    # Plot the array as an image with magma colorbar and title
    fig, ax = plt.subplots()
    cax = ax.imshow(pix_frq_img, cmap='magma')
    fig.colorbar(cax, ax=ax, orientation='vertical')

    # Add title
    ax.set_title('Spike frequency for pixel')

    plt.axis('off')

    # Save the graph
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
def long_term_vis(tensor, filename):
    B, L, D = tensor.shape
    
    # Calculate cosine similarities between sequence's each timesteps
    cosine_similarities = []
    for i in range(L):
        for j in range(i, L):
            similarity = F.cosine_similarity(tensor[:, i, :], tensor[:, j, :], dim=-1).mean().item()
            cosine_similarities.append((i, j, similarity))

    # Cosine similarity
    similarity_matrix = torch.zeros(L, L)
    for i, j, sim in cosine_similarities:
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim

    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Cosine Similarity between Different Time Steps')
    plt.xlabel('Time Step')
    plt.ylabel('Time Step')
    
    # Save as image
    plt.savefig(filename)
    plt.close()
    
    # # Calculate correlation between sequence's each timesteps
    # correlation_matrix = np.zeros((L, L))
    # tensor = tensor.detach().cpu().numpy()

    # for i in range(L):
    #     for j in range(L):
    #         correlation = np.corrcoef(tensor[:, i, :].reshape(-1), tensor[:, j, :].reshape(-1))[0, 1]
    #         correlation_matrix[i, j] = correlation

    # # Vis correlation
    # plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Correlation between Different Time Steps')
    # plt.xlabel('Time Step')
    # plt.ylabel('Time Step')

    # # Save as image
    # plt.savefig(filename)
    # plt.close()
    
def save_BLD_path_as_33grid_image(tensor, index, filename):
    # tensor: [B, L, D]
    tensor = tensor[index]  # [L, D]
    D_idx = [4, 45, 55, 
             100, 142, 165, 
             208, 226, 249]
    
    # Create 3x3 grid fig
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle(filename, fontsize=16)
    axs = axs.flatten()
    
    for i, idx in enumerate(D_idx):
        _seq = tensor[:,idx].detach().cpu().view(32,32).numpy()
        
        # Set _seq as image on the 3x3 grid in order
        axs[i].imshow(_seq, cmap='gray')
        axs[i].set_title(f'Channel idx: {idx}')
        axs[i].axis('off')
        
    # Save the fig
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()