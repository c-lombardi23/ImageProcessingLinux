<h1 align="center">LDC Classifier and Tension Predictor</h1>
<p align="center"><a href="#project-description">Project Description</a> - <a href="#key-features">Key Features</a> - <a href="#technology-stack">Tech Stack</a>
- <a href="#getting-started">Getting Started</a> - <a href="#configuration-file-field-descriptions">Configuration File Field Descriptions</a></p>

## Project Description

This project involves implementing a CNN model to first classify good and bad cleave images uploaded from the user. The images are produced by THORLABS Fiber Cleave Analyzer (FCA) with cleaves produced by the LDC400. If a bad cleave image is obtained, a second, regression-based model is implemented to predict the optimal parameters for producing a good cleave. As of now the model is only capable or predicting tension. A command line interface is provided for training and testing either the CNN model or the regression model. To install, please clone the repository and install the dependencies using the requirements.txt file

## Key Features

This project used a transfer learning model using MobileNetV2 as the base input for the CNN model. After freezing the top layers, I implemented two fully connected layers with parameters determined using Keras Tuner. This was for the image branch of the model. A second branch uses numerical features as input, which is then concatenated with the image branch to produce the full model. For the regression model, I used the dropout layer of the CNN model and then implemented more specific logic to predict the optimal tension.

## Getting Started 

A CLI lets the user train and test the model using different option implemented in a JSON file which is then passed in the CLI through the --file_path option. An example structure of the JSON config file is as follows:

{
  "csv_path": "C:\\Thorlabs\\FCA\\125pm_data.csv", <br>
  "img_folder": "C:\\Users\\clombardi\\125PM", <br>
  "feature_scaler_path": "C:\\Users\\clombardi\\mlp_feature_scaler_6_13.pkl",<br>
  "label_scaler_path": "C:\\Users\\clombardi\\mlp_tension_scaler_6_13.pkl",<br>
  "image_shape": [224, 224, 3],<br>
  "feature_shape": [6],<br>
  "test_size": 0.2,<br>
  "buffer_size":20,<br>
  "batch_size": 4,<br>
  "mode": "train_cnn",<br>
  "learning_rate": 0.01,<br>
  "model_path": "C:\\Users\\clombardi\\125pm_best_mlp_model_6_13.keras",<br>
  "checkpoint_filepath": "",<br>
  "save_history_file":"",<br>
  "save_model_file":"",<br>
  "project_name": "Cleave_Tuner_new1",<br>
  "tuner_directory": "C:\\Users\\clombardi\\tuner1",<br>
  "max_epochs": 3,<br>
  "objective": "val_accuracy",<br>
  "early_stopping": "y",<br>
  "checkpoints":"n",<br>
  "method": "max",<br>
  "monitor":"val_accuracy",<br>
  "patience": 3,<br>
  "img_path": "Fiber-189Plus.png",<br>
  "test_features":[1.6, 17.65, 1, 0, 1]
}

### Configuration File Field Descriptions

Below is a detailed explanation of each key in the JSON configuration file used by the CLI.

| Key | Description |
|-----|-------------|
| `csv_path` | Path to the CSV file containing cleave metadata (e.g., tension, angle, misting, etc.). |
| `img_folder` | Directory containing the cleave images. Filenames must match entries in the CSV. |
| `feature_scaler_path` | Path to the saved `MinMaxScaler` used to scale input features (for classification or regression). |
| `label_scaler_path` | Path to the saved `MinMaxScaler` used to scale tension labels (used for regression model only). |
| `image_shape` | Shape of the input images (e.g., `[224, 224, 3]`). |
| `feature_shape` | Shape of the numerical input features (e.g., `[6]`). |
| `test_size` | Fraction of the dataset to use for testing (e.g., `0.2` = 20%). |
| `buffer_size` | Buffer size used when shuffling the dataset. |
| `batch_size` | Number of samples per training batch. |
| `mode` | Mode of operation. Options: `"train_cnn"` for classification or `"train_regression"` for tension prediction. |
| `learning_rate` | Learning rate for model training. |
| `model_path` | Path to load/save the trained `.keras` model. |
| `checkpoint_filepath` | Optional path to save model checkpoints during training. |
| `save_history_file` | Optional path to save training history as a `.csv` file. |
| `save_model_file` | Optional path to save the trained model. |
| `project_name` | Name of the training project (used in Keras Tuner and logging). |
| `tuner_directory` | Path to save Keras Tuner logs and trial results. |
| `max_epochs` | Maximum number of training epochs. |
| `objective` | Metric to optimize with Keras Tuner (e.g., `"val_accuracy"` or `"val_mae"`). |
| `early_stopping` | Whether to use early stopping (`"y"` or `"n"`). |
| `checkpoints` | Whether to save model checkpoints (`"y"` or `"n"`). |
| `method` | Optimization direction for monitored metric: `"max"` or `"min"`. |
| `monitor` | Metric to monitor during training (e.g., `"val_accuracy"` or `"val_mae"`). |
| `patience` | Number of epochs to wait before early stopping is triggered. |
| `img_path` | Path to a single image to run a prediction on (used for inference mode). |
| `test_features` | List of numerical features to use for a single prediction (e.g., `[angle, scribe_diam, misting, hackle, tearing]`). |


To Install: <br>
pip install git+https://github.com/c-lombardi23/ImageProcessing.git

To Run: <br>
cleave-app --file_path \path\to\config.json

## Tech Stack

*   TensorFlow
*   Keras
*   Keras Tuner
*   Python