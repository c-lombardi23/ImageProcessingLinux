import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

class DataCollector:
  '''
  This class collects data from a csv file and image folder saved in google drive and
  creates datasets for training a machine learning model.
  '''
  def __init__(self, csv_path, img_folder):
    '''
    Parameters:
    -----------------------------------------
    csv_path: str
      - path to csv file in google drive
    img_folder: str
      - path to image folder in google drive
    '''
    if csv_path is None or img_folder is None:
        raise ValueError("Must provide data path")
    self.csv_path = csv_path
    self.img_folder = img_folder
    self.df = self.clean_data()
    self.feature_scaler = None
    self.label_scaler = None

  def set_label(self):
    try:
        df = pd.read_csv(self.csv_path)
    except FileNotFoundError:
        print("CSV file not found!")
        return None

    def label(row):
      if (row['CleaveAngle'] <= 0.45 and not row['Misting'] and not row['Hackle'] and row['ScribeDiameter'] >= 17):
          return "BadDiameter"
      elif (row['CleaveAngle'] <= 0.45 and not row['Misting'] and not row['Hackle']):
          return "Good"
      elif (row['CleaveAngle'] <= 0.45) and (row['Misting'] or row['Hackle']):
          return "Bad_Misting_Hackle"
      else:
          return "BadAngle"


    df["CleaveCategory"] = df.apply(label, axis=1)

    return df

        
  def clean_data(self):
    '''
    Read csv file into dataframe and add column for cleave quality.

    Returns: pandas.DataFrame
      - dataframe with cleave quality column and one-hot encoded labels
    '''
    try:
        df = self.set_label()
    except FileNotFoundError:
        print("CSV file not found!")
        return None

    # Clean image path
    df['ImagePath'] = df['ImagePath'].str.replace(self.img_folder, "", regex=False)

    # One-hot encode CleaveCategory
    ohe = OneHotEncoder()
    onehot_labels = ohe.fit_transform(df[['CleaveCategory']]).toarray()
    class_names = ohe.categories_[0]

    for idx, class_name in enumerate(class_names):
        df[f"Label_{class_name}"] = onehot_labels[:, idx]

    self.encoder = ohe

    return df


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

  def extract_data(self, feature_scaler_path=None):
    '''
    Extract data from dataframe into separate lists for creating datasets.

    Parameters:
    ------------------------------------

    scalar_filename: str
      - path to store pickled scaler 

    Returns: list, list, list
      - lists of images, features, and labels
    '''
    images = self.df['ImagePath'].values
    #features = self.df[['CleaveAngle', 'CleaveTension']].values
    features = self.df[['CleaveAngle', 'CleaveTension', 'ScribeDiameter', 'Misting', 'Hackle', 'Tearing']].values.astype(np.float32)
    label_cols = [col for col in self.df.columns if col.startswith('Label_')]
    labels = self.df[label_cols].values.astype(np.float32)
    self.feature_scaler = MinMaxScaler()
    features = self.feature_scaler.fit_transform(features)
    #joblib.dump(self.scaler, f'./{scaler_filename}.pkl')
    if feature_scaler_path:
      joblib.dump(self.feature_scaler, f'{feature_scaler_path}.pkl')
    return images, features, labels

  def process_images_features(self, inputs, label):
    # Wrapper function for calling image processing
    image_input, features = inputs
    image = self.load_process_images(image_input)
    return (image, features), label
  
  def create_kfold_datasets(self, images, features, labels, buffer_size, batch_size, n_splits=5):
    '''
    Create datasets based on stratified k-fold process for binary classification.

    Parameters:
    --------------------------------------------------------------------

    images: list
      - list of image paths
    features: list
      - list of numerical features
    labels: list
      - list of target values for classification
    buffer_size: int
      - size of buffer to perform shuffling
    batch_size: int
      - size to group data in for training
    n_splits
      - number of k folds

    Returns: list of tuples
      - (train_ds, test_ds)
    
    '''
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=24)

    datasets = []

    for train_index, test_index in kf.split(X=features, y=labels):
      train_imgs, test_imgs = images[train_index], images[test_index]
      train_features, test_features = features[train_index], features[test_index]
      train_labels, test_labels = labels[train_index], labels[test_index]

      train_ds = tf.data.Dataset.from_tensor_slices(((train_imgs, train_features), train_labels))
      test_ds = tf.data.Dataset.from_tensor_slices(((test_imgs, test_features), test_labels))

      train_ds = train_ds.map(lambda x, y: self.process_images_features(x, y))
      test_ds = test_ds.map(lambda x, y: self.process_images_features(x, y))

      train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
      test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

      datasets.append((train_ds, test_ds))

    return datasets
  
  def create_datasets(self, images, features, labels, test_size, buffer_size, batch_size):
    '''
    Creates test and train datasets and splits into different batches after shuffling.

    Parameters:
    -----------------------------------------

    images: list
      - paths to images in google drive
    features: list
      - numerical parameters to label images
    labels: int
      - targets to qualify image quality
    test_size: float
      - decimal between 0 and 1 to represent test size of dataset
    buffer_size: int
      - size of buffer for shuffling data
    batch_size: int
      - size to group data into

    Returns: tf.tensor
      - train and test datasets
    '''
    train_imgs, test_imgs, train_features, test_features, train_labels, test_labels = train_test_split(
        images, features, labels, stratify=labels, test_size=test_size)
    train_ds = tf.data.Dataset.from_tensor_slices(((train_imgs, train_features), train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices(((test_imgs, test_features), test_labels))

    # Map using bound method
    train_ds = train_ds.map(lambda x, y: self.process_images_features(x, y))
    test_ds = test_ds.map(lambda x, y: self.process_images_features(x, y))

    train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds
  
class MLPDataCollector(DataCollector):

    def __init__(self, csv_path, img_folder):
        super().__init__(csv_path, img_folder)
        

    def extract_data(self, feature_scaler_path=None, tension_scaler_path=None):
        '''
        Extract data from dataframe into separate lists for creating datasets.

        Parameters:
        ------------------------------------

        scalar_filename: str
        - path to store pickled scaler 

        Returns: list, list, list
        - lists of images, features, and labels
        '''
        images = self.df['ImagePath'].values
        #features = self.df[['CleaveAngle', 'CleaveTension']].values
        features = self.df[['CleaveAngle', 'ScribeDiameter', 'Misting', 'Hackle', 'Tearing']].values.astype(np.float32)
        labels = self.df['CleaveTension'].values.astype(np.float32)
        self.label_scaler = MinMaxScaler()
        labels = self.label_scaler.fit_transform(labels.reshape(-1, 1))
        self.feature_scaler = MinMaxScaler()
        features = self.feature_scaler.fit_transform(features)
        if feature_scaler_path:
            joblib.dump(self.feature_scaler, f'{feature_scaler_path}.pkl')
        if tension_scaler_path:
            joblib.dump(self.label_scaler, f'{tension_scaler_path}.pkl')
        return images, features, labels
    
    def create_datasets(self, images, features, labels, test_size, buffer_size, batch_size):
        '''
        Creates test and train datasets and splits into different batches after shuffling.

        Parameters:
        -----------------------------------------

        images: list
        - paths to images in google drive
        features: list
        - numerical parameters to label images
        labels: int
        - targets to qualify image quality
        test_size: float
        - decimal between 0 and 1 to represent test size of dataset
        buffer_size: int
        - size of buffer for shuffling data
        batch_size: int
        - size to group data into

        Returns: tf.tensor
        - train and test datasets
        '''
        train_imgs, test_imgs, train_features, test_features, train_labels, test_labels = train_test_split(
            images, features, labels, test_size=test_size)
        train_ds = tf.data.Dataset.from_tensor_slices(((train_imgs, train_features), train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices(((test_imgs, test_features), test_labels))

        # Map using bound method
        train_ds = train_ds.map(lambda x, y: self.process_images_features(x, y))
        test_ds = test_ds.map(lambda x, y: self.process_images_features(x, y))

        train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, test_ds
    
    def create_kfold_datasets(self, images, features, labels, buffer_size, batch_size, n_splits=5):
        '''
        Create datasets based on stratified k-fold process for binary classification.

        Parameters:
        --------------------------------------------------------------------

        images: list
        - list of image paths
        features: list
        - list of numerical features
        labels: list
        - list of target values for classification
        buffer_size: int
        - size of buffer to perform shuffling
        batch_size: int
        - size to group data in for training
        n_splits
        - number of k folds

        Returns: list of tuples
        - (train_ds, test_ds)
        
        '''
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=24)

        datasets = []

        for train_index, test_index in kf.split(images):
            train_imgs, test_imgs = images[train_index], images[test_index]
            train_features, test_features = features[train_index], features[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            train_ds = tf.data.Dataset.from_tensor_slices(((train_imgs, train_features), train_labels))
            test_ds = tf.data.Dataset.from_tensor_slices(((test_imgs, test_features), test_labels))

            train_ds = train_ds.map(lambda x, y: self.process_images_features(x, y))
            test_ds = test_ds.map(lambda x, y: self.process_images_features(x, y))

            train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            datasets.append((train_ds, test_ds))