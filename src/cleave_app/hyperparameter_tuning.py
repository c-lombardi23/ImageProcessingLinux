#import libraries
import tensorflow as tf
from tensorflow.keras.models import  Model, Sequential
from tensorflow.keras.layers import Dense, Concatenate, GlobalAveragePooling2D, Dropout, Input, RandomFlip, RandomRotation, RandomBrightness, RandomZoom, RandomContrast, GaussianNoise
from keras_tuner import HyperModel, Hyperband
from keras.applications import MobileNetV2

class BuildHyperModel(HyperModel):
    '''
    This class build a HyperModel to determine optimal hyperparmeters
    '''
    def __init__(self, image_shape, param_shape):
      '''
      Parameters:
      ----------------------------------------------

      image_shape: tuple
        - dimensions of image
      param_shape: tuple
        - dimensions of parameters
      '''
      self.image_shape = image_shape
      self.param_shape = param_shape

    def build(self, hp):
      '''
      Build hypermodel to perform hyperparameter search.

      Parameters:
      -------------------------

      hp: keras_tuner.engine.hyperparameters.HyperParameters
        - hyperparameters to be tuned
      '''
        # Pre-trained base model
      pre_trained_model = MobileNetV2(
            input_shape=self.image_shape,
            include_top=False,
            weights="imagenet"
        )
      pre_trained_model.trainable = False

        # Data augmentation pipeline
      data_augmentation = Sequential([
            RandomFlip(mode="HORIZONTAL_AND_VERTICAL"),
            RandomRotation(factor=(0.2)),
            RandomBrightness(factor=(0.2)),
            RandomZoom(height_factor=0.1, width_factor=0.1),
            GaussianNoise(stddev=0.01),
            RandomContrast(0.2)
        ])

        # Image input and processing
      image_input = Input(shape=self.image_shape)
      x = data_augmentation(image_input)
      x = pre_trained_model(x, training=False)
      x = GlobalAveragePooling2D()(x)
      x = Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1))(x)

        # Param input and processing
      param_input = Input(shape=self.param_shape)
      y = Dense(
            hp.Int('dense_param1', min_value=16, max_value=128, step=16),
            activation='relu')(param_input)
      y = Dense(
            hp.Int('dense_param2', min_value=8, max_value=64, step=8),
            activation='relu')(y)

        # Combine image and parameter features
      combined = Concatenate()([x, y])

      z = Dense(
            hp.Int('dense_combined', min_value=16, max_value=128, step=16),
            activation='relu')(combined)
      z = Dense(4, activation='softmax')(z)

      model = Model(inputs=[image_input, param_input], outputs=z)

      model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Choice('learning_rate', values=[0.0005, 0.001, 0.01, 0.015])
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

      return model


class HyperParameterTuning:
  '''
  This class is used to tune hyperparameters for model
  '''
  def __init__(self, image_shape, feature_shape, max_epochs=20, objective='val_accuracy', directory='/content/drive/MyDrive/Thorlabs', project_name='Cleave_Tuner3'):
    '''
    Parameters:
    ----------------------------------------------

    image_shape: tuple
      - dimensions of image
    feature_shape: tuple
      - dimensions of parameters
    max_epochs: int
      - maximum number of epochs to train for
      - default: 20
    objective: str
      - metric to monitor during tuning
      - default: val_accuracy
    directory: str
      - directory path to store hyperparameters
      - deafult: /content/drive/MyDrive/Thorlabs
    project_name: str
      - name of project
      - deafult: Cleave_Tuner3

    '''
    self.image_shape = image_shape
    self.feature_shape = feature_shape
    hypermodel = BuildHyperModel(self.image_shape, self.feature_shape)
    self.tuner = Hyperband(
        hypermodel,
        objective=objective,
        max_epochs=max_epochs,
        directory=directory,
        project_name=project_name
    )
  def run_search(self, train_ds, test_ds):
    '''
    Run hyperparameter search

    Parameters:
    ----------------------------------------------

    train_ds: tf.data.Dataset
      - training dataset
    test_ds: tf.data.Dataset
      - testing dataset
      
    '''
  
    self.tuner.search(train_ds, validation_data=test_ds)
  
  def save_best_model(self, pathname):
     best_model = self.get_best_model()
     best_model.save(f"{pathname}.keras")

  def get_best_model(self):
    '''
    Get best model from hyperparameter search

    Returns: tf.keras.Model
      - best model from hyperparameter search
    '''
    return self.tuner.get_best_models(num_models=1)[0]

  def get_best_hyperparameters(self):
    '''
    Get best hyperparameters from hyperparameter search

    Returns: keras_tuner.engine.hyperparameters.HyperParameters
      - best hyperparameters from hyperparameter search
    '''
    return self.tuner.get_best_hyperparameters(num_trials=1)[0]
  
class BuildMLPHyperModel(HyperModel):
    '''
    This class build a HyperModel to determine optimal hyperparmeters
    '''
    def __init__(self, model_path):
        '''
        Parameters:
        -------------------------------------
        model: tf.keras.Model
            - Model to be used for hyperparameter tuning
        '''
        self.cnn_model = tf.keras.models.load_model(model_path)
        self.image_input = self.cnn_model.input[0]
        self.feature_output = self.cnn_model.get_layer('dropout').output

    def build(self, hp):
      '''
      Build model with hyperparameters

      Parameters:
      -------------------------------------
      hp: keras_tuner.HyperParameters
          - Hyperparameters to be used for tuning
      Returns:
      tf.keras.Model
          - Model to be trained
      '''
        # Pre-trained base model

      x = Dense(
            hp.Int('dense_param1', min_value=16, max_value=128, step=16),
            activation='relu')(self.feature_output)
      x = Dense(
            hp.Int('dense_param2', min_value=8, max_value=64, step=8),
            activation='relu')(x)

      feature_input = Input(shape=(5,), name='feature_input')  # Features
        #angle_input = Input(shape=(1,), name='angle_input')  # New input
      y = Dense(
            hp.Int('dense_angle', min_value=16, max_value=128, step=16),
            activation='relu')(feature_input)

      combined = Concatenate()([x, y])
      z = Dense(
            hp.Int('dense_combined', min_value=16, max_value=128, step=16),
            activation='relu')(combined)
      z = Dense(1)(z)

      mlp_hypermodel = Model(inputs=[self.image_input, feature_input], outputs=z)
      mlp_hypermodel.summary()

      mlp_hypermodel.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Choice('learning_rate', values=[0.0005, 0.001, 0.01])
            ),
            loss='mse',
            metrics=['mae']
        )

      return mlp_hypermodel
    
class MLPHyperparameterTuning(HyperParameterTuning):

    def __init__(self, cnn_path, max_epochs=20, objective='val_mae', directory='/content/drive/MyDrive/Thorlabs', project_name='MLPTuner'):
      '''
      Parameters:
      -------------------------------------
      model: tf.keras.Model
          - Model to be used for hyperparameter tuning
      '''
      self.cnn_model = tf.keras.models.load_model(cnn_path)
      self.image_input = self.cnn_model.input[0]
      self.feature_output = self.cnn_model.get_layer('dropout').output
      hypermodel = BuildMLPHyperModel(cnn_path)
      self.tuner = Hyperband(
        hypermodel,
        objective=objective,
        max_epochs=max_epochs,
        directory=directory,
        project_name=project_name
    )