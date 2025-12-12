# Traffic Flow Complex Event Processing (Traffic Flow CEP)

This repository implements a collection of detectors, classifiers and utilities for detecting and analysing congestion events and event chains from traffic sensor data. It contains code for event detection, simple historical baselines, multiple machine learning models, experiments and inference scripts.

## Content

- `data/`: sample datasets, adjacency matrices and event datasets used and produced by provided modules
- `models/`: model definitions and classifier wrappers for TCN, GWNet and DGCRN
- `logs/`: training and experiment logs (per-model run directories), default directory for storing model checkpoints, configs and various results
- `config/`: YAML configuration files for model training/experiments.

## Requirements

Install Python dependencies in `requirements.txt`.
Specific versions of libraries are also provided in the `requirements.txt` if needed

```powershell
python -m pip install -r requirements.txt
```

Recommended Python: 3.11

## Setup / Installation

1. Create a virtual environment and activate it.
2. Install requirements (see above).
3. Confirm `data/` contains the required CSV/pickle files (examples provided in repository).
4. Dasasets METR-LA and PEMS-BAY can be downloaded using the data_utils.py which uses tsl (torch-spatiotemporal),
which is an optional library. Downloaded datasets do not require tsl to load, but can be loaded or stored using csv or pkl. Any dataset can be directly loaded from csv/pickle where it is loaded in the form of a pandas dataframe object where index is the timestamps, and columns is the sensors. This was tested and any dataset in the correct format shuold function.
5. Runnable experiments scripts contain a sample usage, if used independantly

## Data layout

- `data/la.csv`, `data/la.pkl`, `data/la_adj.pkl`: Los Angeles sensor data and adjacency.
- `data/congestion_events_la.csv`: pre-extracted congestion events / chains.
- `data/event_chains.csv`: event chain examples used by chain detector.

If you want to run experiments on your own data, follow the CSV formats used in `data/` as templates.

## Key modules

- `CongestionDetector.py` — single-sensor or multi-sensor congestion detection logic and helpers.
- `CongestionChainDetector.py` — builds chains of congestion events using adjacency/distance heuristics.
- `CongestionEventDataset.py` — dataset class that prepares sliding-window inputs and labels for classifiers.
- `HistoricalAverage.py` — a simple baseline model using historical averages (seasonal baseline).
- `EventClassifier.py` — evaluate and classify congestion events
- `data_utils.py` — data loading, downloading and processing utilities.
- `traffic_classification_experiment.py` — script that runs full train/validation experiments.
- `traffic_inference.py` — script for running inference using a trained model checkpoint.

Models available for use (in `models/`):

- `GWNetClassifier.py` — Graph WaveNet Classification Model
- `DgcrnClassifier.py`— Diffusion Graph Convolutional Recurrent Network Classification Model
- `TCNClassifier.py` — Temporal Convolutional Network Classification Model

## Creating a congesestion event datasets and chains (example)

Firstly you should download the dataset if not downlaoded already.
The distance predefined adjacency matrix can also be downloaded, but is only used when detecting event chains,
so it is optional. 

Simply run congestion detector using its main funtion while specifying the input dataset, and its output will be an events dataset in the data folder. The patterns used to classify and label these events of interested are located inside EventClassifier.py

```powershell
# Generate congestion events dataset using EventClassifier.py classifications 
python CongestionDetector.py
```
The events chain dataset can also be generated, immediately after (uncomment analyze_chains() in the run function)

## Training a model
Training a model requires both the raw dataset and event dataset.
Configs for selecting models, training params and hyper parameters are located inside the yaml configs
as a tree, where `default.yaml` contains the main experiment configureation and model configs contain model specifc hyperparamters and other override options.

Train / run an experiment will use the model and configuration selected in the config/default.yaml and the datasets must be passed to EventsLabelGenerator which will generate the dataset with input output pairs (speed to labels)

```powershell
# Train or run an experiment defined in traffic_classification_experiment.py
python traffic_classification_experiment.py
```

## Running inference (example)

Use `traffic_inference.py` to run a trained checkpoint on test data and produce predictions.
The module will use the defult config located in the same folder as the model checkpoint, so the same parameters used to train the model will be used.

```powershell
# Sample model provided, replace checkpoint with your own model, --dataset and --events_dataset is optional (will defualt to these values)
python traffic_inference.py python "traffic_inference.py logs/dgcrn-2025-12-11_12-12-33/dgcrn.pt" --dataset la.csv --events_dataset data/congestion_events_la.csv
```

Adjust flags (checkpoint path, data path, device) as needed.

## Logs and model checkpoints

- `logs/`: per-run subdirectories (named with model and timestamp). Check for training metrics, saved checkpoints and hyperparameters.
- `models/`: saved model code and helpers — checkpoints may be stored in `logs/` or produced by experiment scripts.

## Tests / Quick verification

There are no formal unit-tests in the repository. To quickly verify things work after setup:

1. Ensure `requirements.txt` is installed.
2. Ensure the required datasets (raw, events, or chains) are downloaded or generated before
3. Run a small experiment or a script that uses the sample data provided.

```powershell
python .\traffic_classification_experiment.py
```
