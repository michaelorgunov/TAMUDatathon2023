import numpy as np
import os
import json
import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Set the desired image size
image_size = (50, 50)

global_max = 1  # Initialize global_max to a reasonable default value

# Function to convert drawing data into an image
def convert_to_image(drawing):
    # Find the maximum values of x and y coordinates
    max_x = max(max(stroke[0]) for stroke in drawing)
    max_y = max(max(stroke[1]) for stroke in drawing)

    # Create a white canvas with a single channel (grayscale)
    image = Image.new("L", image_size, color=255)  # Initialize with white background
    draw = ImageDraw.Draw(image)

    for stroke in drawing:
        for i in range(len(stroke[0]) - 1):
            # Normalize and scale the stroke coordinates to the new image size
            x1 = int(stroke[0][i] * (image_size[0] / max_x))
            y1 = int(stroke[1][i] * (image_size[1] / max_y))
            x2 = int(stroke[0][i + 1] * (image_size[0] / max_x))
            y2 = int(stroke[1][i + 1] * (image_size[1] / max_y))
            draw.line([(x1, y1), (x2, y2),], fill=0, width=2)  # Draw in black

    return image

# Function to normalize data
def normalizeData(data, max_value):
    data[..., :2] /= max_value

# Function to parse a single line from the ndjson file
def parseLine(ndjson_line):
    data = json.loads(ndjson_line)
    name = data["word"]
    drawing = data["drawing"]
    image = convert_to_image(drawing)  # Convert drawing data to image
    return name, image

# Function to read lines from the ndjson file
def readFile(ndjson_file, start, end):
    names = []
    image_data = []
    with open(ndjson_file, "r") as file:
        counter = 0
        for line in file:
            if counter == end:
                break
            if counter >= start:
                line_name, line_data = parseLine(line)
                if (0 <= counter <= 10):
                    line_data.save(f"../data/images/image{counter}.png")
                names.append(line_name)
                image_data.append(line_data)
            counter += 1
    return names, image_data

def main():
    # Set the maximum number of files and lines to process
    max_files = 3
    max_lines = 2000
    temp = []
    total_data = []
    total_names = []
    file1_data = []
    file2_data = []
    file1_names = []
    file2_names = []

    if len(sys.argv) > 1:
        input_directory = sys.argv[1]

        for filename in os.listdir(input_directory):
            if max_files == 0:
                break

            if filename.endswith(".ndjson"):
                ndjson_file = os.path.join(input_directory, filename)
                print(f"Processing file: {ndjson_file}")

                names, temp = readFile(ndjson_file, 0, max_lines)
                half = len(temp) // 2
                print("half ", half)
                # Grab values for the first and second files
                file1_data.extend(temp[:half])
                file2_data.extend(temp[half:])

                file1_names.extend(names[:half])
                file2_names.extend(names[half:])

                total_data.extend(temp)
                total_names.extend(names)

                max_files -= 1

        if total_data:
            # Create an empty array to hold the image data with padding
            padded_data_1 = np.zeros((len(file1_data), image_size[0], image_size[1]), dtype=np.float32)
            padded_data_2 = np.zeros((len(file2_data), image_size[0], image_size[1]), dtype=np.float32)

            # fill the zeros with the data in the array
            if total_data:
                for i in range(len(file1_data)):
                    padded_data_1[i] = np.array(file1_data[i])

                for i in range(len(file2_data)):
                    padded_data_2[i] = np.array(file2_data[i])
            # Output file names for both sets of lines
            output_file_name_1 = os.path.join("../data/npzShort/combined_data_1.npz")
            output_file_name_2 = os.path.join("../data/npzShort/combined_data_2.npz")

            # Save the preprocessed data for the first and second sets of lines into separate output files
            np.savez_compressed(output_file_name_1, x=padded_data_1, y=file1_names)
            np.savez_compressed(output_file_name_2, x=padded_data_2, y=file2_names)

            print(f"Preprocessed data for the first set saved to {output_file_name_1}")
            print(f"Preprocessed data for the second set saved to {output_file_name_2}")

        else:
            print("No data to process.")
    else:
        print("No directory provided. Please provide a directory containing .ndjson files as a command line argument.")
        sys.exit(1)

if __name__ == "__main__":
    main()
