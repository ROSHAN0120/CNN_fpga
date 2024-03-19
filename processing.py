import cv2
import os
import numpy as np

def load_data():
    # Path to the folder containing the images
    folder_path = 'E:/Mini_Project/cyclone_cn/insat3d_ir_cyclone_ds/CYCLONE_DATASET_INFRARED/'

    # Path to save processed images
    # output_folder = 'processed_images/'

    # Create the output folder if it doesn't exist
    # os.makedirs(output_folder, exist_ok=True)

    # List to store normalized image values
    training_input = []

    # Iterate over each image file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            # Read the color image
            color_image = cv2.imread(os.path.join(folder_path, filename))

            # Convert the color image to grayscale
            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Resize the grayscale image to 28x28 pixels
            resized_image = cv2.resize(grayscale_image, (28, 28))

            # Normalize pixel values to the range [0, 1]
            normalized_image = resized_image / 255.0

            # Flatten the image to a 1D array
            flattened_image = normalized_image.reshape(-1)

            # Save the resized image
            #resized_output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.jpg')
            # cv2.imwrite(resized_output_path, resized_image)

            # Append normalized image to the training input list
            training_input.append(flattened_image)

            #print(f"Processed image: {filename}")
            #print("Values:", normalized_image)
            #print("Resized image shape:", resized_image.shape)
            #print("Flattened image shape:", flattened_image.shape)

    # Convert the list of normalized images to a numpy array
        training_input = np.array(training_input)
        training_results = [25, 27, 28, 30, 30, 31, 32, 32, 33, 33, 33, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 38, 39, 40, 40, 40, 40, 41, 42, 42, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 45, 46, 46, 46, 47, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 52, 52, 53, 53, 53, 53, 54, 55, 55, 56, 57, 57, 57, 58, 58, 59, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 67, 67, 68, 69, 69, 70, 73, 74, 74, 74, 75, 77, 77, 81, 81, 82, 82, 83, 84, 84, 85, 85, 85, 86, 86, 86, 87, 94, 98, 99, 101, 102, 106, 111, 112, 115, 118, 119, 128]

    # Check the shape of the training input array
        print("Shape of training input array:", training_input[0].shape)
        print("Shape of training input array:", training_results[0])

        training_data = zip(training_input, training_results)
        return(training_data)
