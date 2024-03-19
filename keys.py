import h5py

# Open the provided file
with h5py.File('E:/Mini_Project/cyclone_cn/Model.h5', 'r') as f:
    # Create a text file to write weights and biases
    with open('weights_and_biases.txt', 'w') as file:
        # Write the layer name to the file
        file.write('Layer: model_weights\n\n')
        
        # Iterate over the keys of the 'model_weights' layer
        for key in f['model_weights'].keys():
            # Write the key (layer name) to the file
            file.write(f'Layer: {key}\n')
            
            # Iterate over the keys of each layer
            for subkey in f[f'model_weights/{key}'].keys():
                # Check if the key contains weight or bias data
                if 'kernel' in subkey or 'bias' in subkey:
                    # Write the weight or bias name
                    file.write(f'{subkey}:\n')
                    
                    # Get the data and write it to the file
                    data = f[f'model_weights/{key}/{subkey}'][:]
                    for d in data:
                        file.write(f'{d}\n')
                    file.write('\n')
