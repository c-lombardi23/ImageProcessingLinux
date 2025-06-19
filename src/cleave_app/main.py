from .config_schema import Config
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


from .data_processing import *
from .model_pipeline import *
from .prediction_testing import *
from .hyperparameter_tuning import *

import argparse
import json


def load_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return Config(**data)

def train_cnn(config):
    data = DataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data(config.feature_scaler_path)
    train_ds, test_ds = data.create_datasets(images, features, labels, config.test_size, config.buffer_size, config.batch_size)
    trainable_model = CustomModel(train_ds, test_ds)
    compiled_model = trainable_model.compile_model(config.image_shape, config.feature_shape, config.learning_rate)
    if config.checkpoints == "y":
        checkpoint = trainable_model.create_checkpoints(config.checkpoint_filepath, config.monitor, config.method)
    else:
        checkpoint = None
    if config.early_stopping == "y":
        es = trainable_model.create_early_stopping(config.patience, config.method, config.monitor)
    else:
        es = None
    if config.max_epochs == None:
        config.max_epochs = 20
    history = trainable_model.train_model(compiled_model, epochs=config.max_epochs, early_stopping=es, checkpoints=checkpoint, history_file=config.save_history_file, model_file=config.save_model_file)
    trainable_model.plot_metric("Loss vs. Val Loss", history.history['loss'], history.history['val_loss'], 'loss', 'val_loss', 'epochs', 'loss')
    trainable_model.plot_metric("Accuracy vs. Val Accuracy", history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 'val_accuracy', 'epochs', 'accuracy')


def train_mlp(config):
    data = DataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data(config.feature_scaler_path)
    train_ds, test_ds = data.create_datasets(images, features, labels, config.test_size, config.buffer_size, config.batch_size)
    trainable_model = BuildMLPModel(config.model_path, train_ds, test_ds)
    compiled_model = trainable_model.compile_model(config.feature_shape)
    if config.checkpoints == "y":
        checkpoint = trainable_model.create_checkpoints(config.checkpoint_filepath, config.monitor, config.method)
    else:
        checkpoint = None
    if config.early_stopping == "y":
        es = trainable_model.create_early_stopping(config.patience, config.method, config.monitor)
    else:
        es = None
    if config.max_epochs == None:
        config.max_epoch = 20
    history = trainable_model.train_model(compiled_model, epochs=config.max_epochs, early_stopping=es, checkpoints=checkpoint, history_file=config.save_history_file, model_file=config.save_model_file)
    trainable_model.plot_metric("Loss vs. Val Loss", history.history['loss'], history.history['val_loss'], 'loss', 'val_loss', 'epochs', 'loss')
    trainable_model.plot_metric("Accuracy vs. Val Accuracy", history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 'val_accuracy', 'epochs', 'accuracy')

def train_kfold_cnn(config):
    data = DataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data(config.feature_scaler_path)
    datasets = data.create_kfold_datasets(images, features, labels, config.buffer_size, config.batch_size)
    k_models, kfold_histories = CustomModel.train_kfold(datasets, config.image_shape, config.feature_shape, config.learning_rate, history_file = config.save_history_file,
                                            model_file = config.save_model_file)
    CustomModel.get_averages_from_kfold(kfold_histories)


def train_kfold_mlp(config):
    data = MLPDataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data(config.feature_scaler_path)
    datasets = data.create_kfold_datasets(images, features, labels, config.buffer_size, config.batch_size)
    k_models, kfold_histories = CustomModel.train_kfold(datasets, config.image_shape, config.feature_shape, config.learning_rate, metrics=['mae'], history_file = config.save_history_file,
                                            model_file = config.save_model_file)
    CustomModel.get_averages_from_kfold(kfold_histories)


def run_search_helper(config, tuner, train_ds, test_ds):
    tuner.run_search(train_ds, test_ds)
    print(tuner.get_best_hyperparameters().values)
    pathname = config.best_model_path
    if pathname == None:
        print("Model not saved")
        exit()
    else:
        tuner.save_best_model(pathname)
        print(f"Model saved to: {pathname}")

def cnn_hyperparameter(config):
    data = DataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data(config.feature_scaler_path)
    train_ds, test_ds = data.create_datasets(images, features, labels, config.test_size, config.buffer_size, config.batch_size)
    if config.max_epoch == None:
        max_epoch = 20
    tuner = HyperParameterTuning(config.image_shape, config.feature_shape, max_epochs=max_epoch, project_name=config.project_name, directory=config.tuner_directory)
    run_search_helper(config, tuner, train_ds, test_ds)
  
def mlp_hyperparameter(config):
    data = MLPDataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data(config.feature_scaler_path)
    train_ds, test_ds = data.create_datasets(images, features, labels, config.test_size, config.buffer_size, config.batch_size)
    if config.max_epoch == None:
        max_epoch = 20
    tuner = MLPHyperparameterTuning(config.image_shape, config.feature_shape, max_epochs=max_epoch, project_name=config.project_name, directory=config.tuner_directory)
    run_search_helper(config, tuner, train_ds, test_ds)

def test_cnn(config):
    tester = TestPredictions(config.model_path, config.csv_path, config.feature_scaler_path, config.img_folder)
    pred_labels, predictions = tester.gather_predictions()
    tester.display_confusion_matrix(pred_labels)
    tester.display_classification_report(tester.true_labels, pred_labels)

def test_mlp(config):
    test_model = tf.keras.models.load_model(config.model_path)
    tester = TensionPredictor(test_model, config.img_folder, config.img_path, config.label_scaler_path, config.feature_scaler_path)
    predicted_tension = tester.PredictTension(config.test_features)
    print(f"Predicted Tension: {predicted_tension:.0f}g")
 

def choices(mode, config):
    if mode == "train_cnn":
        train_cnn(config)
    elif mode == "train_mlp":
        train_mlp(config)
    elif mode == "cnn_hyperparameter":
        cnn_hyperparameter(config)
    elif mode == "mlp_hyperparameter":
        mlp_hyperparameter(config)
    elif mode == "test_cnn":
        test_cnn(config)
    elif mode == "test_mlp":
        test_mlp(config)
    elif mode == "train_kfold_cnn":
        train_kfold_cnn(config)
    elif mode == "train_kfold_mlp":
        train_kfold_mlp(config)

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="Train Model from command line")
    parser.add_argument("--file_path", required=True)
    parsed_args = parser.parse_args(args)

    filepath = parsed_args.file_path
    config = load_file(filepath)
    mode = config.mode
    choices(mode, config)


if __name__ == "__main__":
    main()  

'''def main(args):
    filepath = args.file_path
    config = load_file(filepath)
    mode = config.mode
    choices(mode, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model from command line")
    parser.add_argument("--file_path", required=True)
    args = parser.parse_args()
    main(args)
    '''