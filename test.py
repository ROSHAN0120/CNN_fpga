from keras.models import load_model
model = load_model('data/weights-improvement-446-0.00.hdf5')
for layer in model.layers:
     if len(layer.weights) > 0:
        print(layer.name, layer.weights[0].shape)