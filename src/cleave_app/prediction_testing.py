#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler



class TestPredictions:
  '''
  This class is used to test model performance on unseen data using metrics such as
  accuracy, precision, recall, and confusion matrix.
  '''
  def __init__(self, model_path, csv_path, scalar_path, img_folder):
    '''
    Parameters:
    ----------------------------------------------

    model_path: str
      - path to model in google drive
    csv_path: str
      - path to csv file in google drive
    scalar_path: str
      - path to scaler in google drive
    '''
    self.scalar_path = scalar_path
    self.img_folder = img_folder
    self.model = tf.keras.models.load_model(model_path)
    self.csv_path = csv_path

  def clean_data(self):
    '''
    Read csv file into dataframe and add column for cleave quality.

    Returns: pandas.DataFrame
      - dataframe with cleave quality column
    '''
    try:
      df = pd.read_csv(self.csv_path)
    except FileNotFoundError:
      print("File not found")
      return None
    df['CleaveQuality'] = ((df['CleaveAngle'] <= 0.45) & (df['Misting'] == 0) & (df['Hackle'] == 0)).astype(int)
    # Clean image path to read from google drive
    df['ImagePath'] = df['ImagePath'].str.replace(self.img_folder, "")
    print(df)    
    pred_image_paths = df['ImagePath'].values
    pred_features = df[['CleaveAngle', 'CleaveTension', 'ScribeDiameter', 'Misting', 'Hackle', 'Tearing']].values.astype(np.float32)
    self.true_labels = list(df['CleaveQuality'])
    return pred_image_paths, pred_features

  def load_process_images(self, filename):
    '''
    Load image from path in google drive and standardize to 224x224.

    Parameters:
    -----------------------------------------
    filename: str
      - path to image in google drive

    Returns: tf.tensor
      - image in tensor format
    '''
    def _load_image(file):
      file = file.numpy().decode('utf-8')
      full_path = os.path.join(self.img_folder, file)
      try:
        img_raw = tf.io.read_file(full_path)
      except FileNotFoundError:
        print("File not found")
        return None
      img = tf.image.decode_png(img_raw, channels=1)
      img = tf.image.resize(img, [224, 224])
      img = tf.image.grayscale_to_rgb(img)
      img = img / 255.0
      return img

    img = tf.py_function(_load_image, [filename], tf.float32)
    img.set_shape([224, 224, 3])
    return img

  def test_prediction(self, image_path, feature_vector):
    '''
    Test function for generating prediction

    Parameters:
    ----------------------------------------------

    image_path: str
      - path to image to predict
    tension: int
      - tension value in grams
    cleave_angle: float
      - angle that was achieved from cleave

    Return: tf.keras.Model
      - predicition from new image of good or bad cleave
    '''
    image = self.load_process_images(image_path)
    image = np.expand_dims(image, axis=0)

    scalar = joblib.load(self.scalar_path)
    scaled_features = scalar.transform([feature_vector]) 

    prediction = self.model.predict([image, scaled_features])
    return prediction

  def gather_predictions(self):
    '''
    Gather multiple predictions from test data

    Returns: list
      - list of predictions
    '''

    pred_image_paths, pred_features = self.clean_data()
    predictions = []
    for img_path, feature_vector in zip(pred_image_paths, pred_features):
      prediction = self.test_prediction(img_path, feature_vector)
      predictions.append(prediction)

    # Set prediction labels to 0 or 1 based on probability
    pred_labels = [1 if pred[0][0] > 0.5 else 0 for pred in predictions]
    return pred_labels, predictions

  def display_confusion_matrix(self, pred_labels):
    '''
    Displays confusion matrix metric comparing true labels to predicted labels.

    Parameters:
    ----------------------------------------------

    pred_labels: list
      - list of predicted labels
    '''
    cm = confusion_matrix(self.true_labels, pred_labels, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Bad Cleave', 'Good Cleave'])
    disp.plot()
    plt.show()

  def display_classification_report(self, true_labels, pred_labels):
    '''
    Diplays classification report comparing true labels to predicted labels.

    Parameters:
    ----------------------------------------------

    true_labels: list
      - list of true labels
    pred_labels: list
      - list of predicted labels
    '''
    print(classification_report(true_labels, pred_labels))

  def plot_roc(self, title, true_labels, pred_probabilites):
    pred_probabilites = np.array(pred_probabilites).flatten()
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probabilites)

    auc = roc_auc_score(true_labels, pred_probabilites)

    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc:.2f}%)')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

class TensionPredictor:

    def __init__(self, model, image_folder, image_path, tension_scaler_path, feature_scaler_path):
        self.model = model
        self.image_path = image_path
        self.image_folder = image_folder
        self.tension_scaler = joblib.load(tension_scaler_path)
        self.feature_scaler = joblib.load(feature_scaler_path)

    def load_and_preprocess_image(self, file_path, img_folder):
        '''
        Load and preprocess image from file path

        Parameters:
        -------------------------------------
        file_path: str
            - path to image file
        img_folder: str
            - path to image folder
            Returns:
        tf.Tensor
            - preprocessed image
        '''
        # Construct full path
        full_path = os.path.join(img_folder, file_path)
        img_raw = tf.io.read_file(full_path)
        img = tf.image.decode_png(img_raw, channels=1)
        img = tf.image.resize(img, [224, 224])
        img = tf.image.grayscale_to_rgb(img)
        # Normalize image
        img = img / 255.0
        return img

    def PredictTension(self, features):
        '''
        Predict tension for given image and angle

        Parameters:
        -------------------------------------
        model: tf.keras.Model
            - Model to be used for prediction
        image_path: str
            - Path to image to be used for prediction
            angle: float
            - Angle to be used for prediction

        Returns:
        float
            - Predicted tension
        '''
        # Process image and convert angle and image to tensor with dimension for single batch
        image = self.load_and_preprocess_image(self.image_path, self.image_folder)
        image = tf.expand_dims(image, axis=0)
        features = np.array(features).reshape(1, -1)
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        # Predict tension
        features = self.feature_scaler.fit_transform(features)
        predicted_tension = self.model.predict([image, features])
        # Scale tension back to normal units
        predicted_tension = self.tension_scaler.inverse_transform(predicted_tension)
        # Print tensions
        return predicted_tension[0][0]

    def plot_metric(self, title, X, y, x_label, y_label, x_legend, y_legend):
        '''
        Plot metric

        Parameters:
        -------------------------------------
        title: str
            - Title of plot
        X: list
            - List of x values
        y: list
            - List of y values
            x_label: str
            - Label for x axis
        y_label: str
            - Label for y axis
        x_legend: str
            - Legend for x axis
        y_legend: str
            - Legend for y axis
        '''
        plt.title(title)
        plt.plot(X, label=x_legend)
        plt.plot(y, label=y_legend)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='lower right')
        plt.show()