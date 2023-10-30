
# TAMUDatathon2023

## Extracting and Preparing Data

1.  ```cd`backend/dataParse``` 
2. ```python3 shortDataParser.py ../data/ndjson``` Change ndjson with desired directory

## Generating and Training the Model
1.  ```cd`backend/modelTraining``` 
2. ```python3 sequentialModelTrainer.py ../data/npzShort/combined_data_1.npz ../data/npzShort/combined_data_2.npz```  Change npz files as needed
## Tester Files
1.  ```cd`backend/tester``` 
2. ```gpuDevices.py``` Displays list of devices
3. ```npviewer.py``` Displays shape and sample data