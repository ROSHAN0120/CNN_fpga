import h5py

# Path to the model weights file
weights_file_path = 'E:/    Mini_Project/cyclone_cn/Model.h5'

# Open the model weights file
with h5py.File(weights_file_path, 'r') as f:
    # Iterate over the keys in the file
    for layer_name in f.keys():
        # Check if the layer has weights and biases
        if layer_name.startswith('conv2d_') or layer_name.startswith('dense_'):
            # Get the weights and biases
            weights = f[layer_name][layer_name]['kernel:0'][:]
            biases = f[layer_name][layer_name]['bias:0'][:]
            
            # Print the layer name, weights, and biases
            print(f'Layer: {layer_name}')
            print('Weights:')
            print(weights)
            print('Biases:')
            print(biases)
            print()
