import os
import h5py
import numpy as np

def print_model_h5_wegiths(weight_file_path):
    # weights tensor is stored in the value of the Dataset, and each episode will have attrs to store the attributes of each network layer

    f = h5py.File(weight_file_path) # read weights h5 file and return File class
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(f.filename)) # weight_file_path
            print("Root attributes:")
        for key, value in f.attrs.items():
            print(" {}: {}".format(key, value))
            # Output the attrs information stored in the File class, generally the name of each layer: layer_names/backend/keras_version

        for layer, g in f.items():
            # Read the name of each layer and the Group class containing layer information
            print(" {} with Group: {}".format(layer, g)) # model_weights with Group: <HDF5 (22 members)>),
            print(" Attributes:")
            for key, value in g.attrs.items():
                # Output the attrs information stored in the Group class, generally the weights and biases of each layer and their names
                # eg ;weight_names: [b'attention_2/q_kernel:0' b'attention_2/k_kernel:0' b'attention_2/w_kernel:0']
                print(" {}: {}".format(key, value))
                #
                print(" Dataset:") # np.array(f.get(key)).shape()
            for name, d in g.items(): # Read the Dataset class that stores specific information in each layer
                print('name:', name, d)

                if str(f.filename).endswith('.weights'):
                    for k, v in d.items():
                        # Output the layer name and weight stored in the Dataset, or print the attrs of the dataset
                        # k, v embeddings:0 <HDF5 dataset "embeddings:0": shape (21, 128), type "<f4">
                        print(' {} with shape: {} or {}'.format(k, np.array(d.get(k)).shape, np.array(v).shape))
                        print(" {} have weights: {}".format(k, np.array(v))) # Weights of each layer
                        print(str(k))
                if str(f.filename).endswith('.h5'):
                    for k, v in d.items(): # v is equivalent to d.get(k)
                        print(k, v)
                        print(' {} with shape: {} or {}'.format(k, np.array(d.get(k)).shape, np.array(v).shape))
                        print(" {} have weights: {}".format(k, np.array(v))) # Weights of each layer
                        print(str(k))

                        # Adam <HDF5 group "/optimizer_weights/training/Adam" (63 members)>

    finally:
        f.close()

print('Current working path:', os.getcwd())
h5_weight = r'Model.h5'
print_model_h5_wegiths(h5_weight)   