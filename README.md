# federated-learning-DIA

This repository contains the source code of the paper **Robust Federated Learning based on Metrics Learning and Unsupervised Clustering for Malicious Data Detection**

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Training the Encoder Network

Under subfolder *cloud*, there are two folders *cifar* and *encoder* which performs experiment on dataset CIFAR-10 and Fashion-MNIST respectively. Two experiments has almost the same structure. Let's take *encoder* as an example.

You can customize the Encoder Network structure, training hyper parameters, and training the encoder model  in `metrics_leaning_model.py`, but be sure to keep consistent model structure in `encoder.py`. There is one pre-trained model weight saved under *weights* folder.

## K-Means Clustering

The K-means clustering and data cleaning process is performed by the function in `kmeans.py`.  

## Adjust Local Training Epoch and Total Communication Rounds

Local training epoch is adjusted in file `client.py`, in **fit()** function of **Client** class.

Communication round can be customized in last line of file `server.py`

## Config and Run Experiment

In `main.py`, you can change experiment parameters

```python
CLIENT_NUMBER = 10 # Total Number of Clients
FAULT_INDEX = 7 # FAULT_INDEX = number_of_corrupt_client - 1
# For example, when there is no corrupt clients, FAULT_INDEX = -1, when there are 8 corrupt clients, FAULT_INDEX = 7
FAULT_RATIO = 0.8 # Fraction of data are mislabed in corrupted clients
IS_KMEANS = True # If use proposed method
```

To run the experiment, simply run (on Linux)

```bash
nohup python main.py &&
```

The main function will first split the dataset for clients, then it runs server and clients on multiple processes.While it running, it will create fodder under *logs*. When Kmeans is enable, the experiment result will be saved in folder `{CLIENT_NUMBER}_{FAULT_INDEX+1}_attack_{FAULT_RATIO}_kmeans`, if Kmeans is not enable, the experiment result will be saved in folder `{CLIENT_NUMBER}_{FAULT_INDEX+1}_attack_{FAULT_RATIO}`

## Visualize Experiment Results

You can find the evaluation result in `server.log` under the *logs/{experiment_name}*. Add data in `plot_acc.py` or `plot_multi_acc.py` to get visualized experiment results.







