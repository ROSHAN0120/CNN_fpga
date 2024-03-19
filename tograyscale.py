import cv2

# Read the color image
color_image = cv2.imread('E:/Mini_Project/cyclone_cn/insat3d_ir_cyclone_ds/CYCLONE_DATASET_INFRARED/25.jpg')

# Convert the color image to grayscale
grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Save or display the grayscale image
cv2.imwrite('grayscale_image.jpg', grayscale_image)
# To display the image:
cv2.imshow('Grayscale Image', grayscale_image)
cv2.waitKey(0)
#cv2.destroyAllWindows()
print(grayscale_image.shape)


#reshape 
grayscale_image = cv2.imread('grayscale_image.jpg', cv2.IMREAD_GRAYSCALE)

# Resize the image to 28x28 pixels
resized_image = cv2.resize(grayscale_image, (28, 28))

# Flatten the image to a 1D array
flattened_image = resized_image.reshape(-1)
# Save the resized image
cv2.imwrite('resized_image.jpg', resized_image)

# Convert pixel values to the range [0, 1]
normalized_image = flattened_image / 255.0

# Example of accessing pixel values:
# Access pixel at row=0, column=0

print("values:", normalized_image)
print(resized_image.shape)
print(flattened_image.shape)


#show

#cv2.imshow('Image', resized_image)
#cv2.waitKey(0)