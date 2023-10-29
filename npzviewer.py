import numpy as np

# Load the NPZ file
data = np.load('npzShort/combined_data_1.npz')

# Access the data
x_data = data['x']
y_data = data['y']

# Perform operations on the data
print("Shape of x_data:\n", x_data[0])
print("Shape of y_data:", y_data)
#print(np.where(y_data == 'aircraft carrier')[0])

print(data.files)
print(x_data.shape)
print(y_data.shape)


# Close the NPZ file (optional)
data.close()




