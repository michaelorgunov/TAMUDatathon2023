import numpy as np
import os
import json
import sys

global_max = 0

# Turn a JSON object into usable data
def parseLine(ndjson_line):
    data = json.loads(ndjson_line)
    name = data["word"]
    drawing = data["drawing"]
    stroke_lengths = [len(stroke[0]) for stroke in drawing]
    total_points = sum(stroke_lengths)
    np_data = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    max_x, max_y = 0, 0  # Initialize max_x and max_y

    for stroke in drawing:
        for i in [0, 1]:
            # Convert integer coordinates to floats
            stroke_coords = np.array(stroke[i], dtype=np.float32)
            np_data[current_t:(current_t + len(stroke_coords)), i] = stroke_coords

            # Update max_x and max_y
            max_x = max(max_x, np.max(stroke_coords))
            max_y = max(max_y, np.max(stroke_coords))

        current_t += len(stroke_coords)
        np_data[current_t - 1, 2] = 1  # stroke_end

    global global_max
    global_max = max(global_max, max_x, max_y)

    return name, np_data

def normalizeData(data, max_value):
    #data[:, 0:2] /= max_value
    print(1)

# Read the NDJSON file
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
                file1_data.extend(temp[:half])
                file2_data.extend(temp[half:])

                file1_names.extend(names[:half])
                file2_names.extend(names[half:])

                total_data.extend(temp)
                total_names.extend(names)

                max_files -= 1

        if total_data:
            # Find the maximum number of points among all arrays in total_data
            max_points = max(arr.shape[0] for arr in total_data)

            # Create an empty array to hold the data with padding
            padded_data_1 = np.zeros((len(file1_data), max_points, 3), dtype=np.float32)
            padded_data_2 = np.zeros((len(file2_data), max_points, 3), dtype=np.float32)

            # Fill the padded_data array with the data from total_data
            for i, arr in enumerate(file1_data):
                padded_data_1[i, :arr.shape[0], :] = arr

            for i, arr in enumerate(file2_data):
                padded_data_2[i, :arr.shape[0], :] = arr

            # Normalize the data using global max values after processing both sets of lines
            normalizeData(padded_data_1, global_max)
            normalizeData(padded_data_2, global_max)

            # Output file names for both sets of lines
            output_file_name_1 = os.path.join("npzShort", "combined_data_1.npz")
            output_file_name_2 = os.path.join("npzShort", "combined_data_2.npz")

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
