#import split_data
import network2
import cyclone_provider

# Assume 'data' is your dataset
# data = cyclone_loader.load_data()
#dataset = data[0]
# data_list = list(data)


training_data,validation_data = cyclone_provider.prepare_training_data()
net = network2.Network([102400,500, 30, 30, 128])#change the number of layers or number of neurons in each layer here
validation_data = list(validation_data)
# print(validation_data[0])
training_data = list(training_data)
net.SGD(training_data, 50, 2, .1, lmbda=5.0,evaluation_data=validation_data, monitor_evaluation_accuracy=True)
net.save("WeigntsAndBiasesCyclone.txt") 