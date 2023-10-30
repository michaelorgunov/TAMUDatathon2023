import numpy as np
import os
import json
import sys
from PIL import Image, ImageDraw

# Turn a JSON object into usable data
def parseLine(ndjson_line):
    data = json.loads(ndjson_line)
    name = data["word"]
    drawing = data["drawing"]
    strokeLengthsArr = [len(stroke[0]) for stroke in drawing]
    totPoints = sum(strokeLengthsArr)
    npData = np.zeros((totPoints, 3), dtype=np.float32)
    current_t = 0
    for stroke in drawing:
        for i in [0, 1]:
            npData[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        npData[current_t - 1, 2] = 1  # stroke_end
    # Preprocessing.
    # 1. Size normalization.
    lower = np.min(npData[:, 0:2], axis=0)
    upper = np.max(npData[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    npData[:, 0:2] = (npData[:, 0:2] - lower) / scale
    # 2. Compute deltas.
    npData[1:, 0:2] -= npData[0:-1, 0:2]
    npData = npData[1:, :]

    return name, npData

# Read the NDJSON file
def readFile(ndjson_file):
    names = []
    imageData = []
    with open(ndjson_file, "r") as file:
        for line in file:
            lineName, lineData = parseLine(line)
            names.append(lineName)
            imageData.append(lineData)
    return names, imageData

def main():
    if len(sys.argv) > 1:
        input_directory = sys.argv[1]

        # Create the output directory if it doesn't exist
        output_directory = "npz"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for filename in os.listdir(input_directory):
            if filename.endswith(".ndjson"):
                ndjson_file = os.path.join(input_directory, filename)
                print(f"Processing file: {ndjson_file}")

                names, imageData = readFile(ndjson_file)

                # Find the maximum number of points among all arrays
                max_points = max(arr.shape[0] for arr in imageData)

                # Create an empty array to hold the data with padding
                padded_data = np.zeros((len(imageData), max_points, 3), dtype=np.float32)

                # Fill the padded_data array with the data from imageData
                for i, arr in enumerate(imageData):
                    padded_data[i, :arr.shape[0], :] = arr

                # Save the preprocessed data to the output directory
                output_file_name = os.path.join(output_directory, filename.replace(".ndjson", ".npz"))
                np.savez_compressed(output_file_name, x=padded_data, y=names)

                print(f"Preprocessed data saved to {output_file_name}")
    else:
        print("No directory provided. Please provide a directory containing .ndjson files as a command line argument.")
        sys.exit(1)

if __name__ == "__main__":
    main()
