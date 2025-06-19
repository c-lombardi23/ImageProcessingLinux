from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
import os

class Config(BaseModel):
    # required inputs
    csv_path: str
    img_folder: str
    mode: str
    image_shape: List[int]
    feature_shape: List[int]

    # optional inputs
    feature_scaler_path: Optional[str] = None 
    label_scaler_path: Optional[str] = None
    model_path: Optional[str] = None
    learning_rate: Optional[float] = 0.001
    buffer_size: Optional[int] = 32
    batch_size: Optional[int] = 8
    test_size: Optional[float] = 0.2
    img_path: Optional[str] = None
    test_features: Optional[List[float]] = None
    max_epochs: Optional[int] = None
    early_stopping: Optional[str] = "n"
    tuner_directory: Optional[str] = None
    checkpoints: Optional[str] = "n"
    objective: Optional[str] = "val_accuracy"
    method: Optional[str] = "max"
    patience: Optional[int] = 3
    checkpoint_filepath: Optional[str] = None
    monitor: Optional[str] = "val_accuracy"
    project_name: Optional[str] = None
    save_model_file: Optional[str] = None
    save_history_file: Optional[str] = None
    best_model_path: Optional[str] = None

    @field_validator("csv_path", "img_folder", mode="before")
    @classmethod
    def path_exists(cls, value):
        if not os.path.exists(value):
            raise ValueError(f"{value} does not exist!")
        return value

    @field_validator("mode")
    @classmethod
    def valid_modes(cls, value):
        valid_modes = [
            'train_cnn', 'train_mlp',
            'cnn_hyperparamter', 'mlp_hyperparameter',
            'test_cnn', 'test_mlp', 'train_kfold_cnn', 'train_kfold_mlp'
        ]
        if value not in valid_modes:
            raise ValueError(f"{value} is not a valid mode!")
        return value
    
    @model_validator(mode="after")
    def valid_shapes(self):
        if self.mode in {"train_cnn", "cnn_hyperparameter"}:
            if self.feature_shape != [6]:
                raise ValueError("Feature shape must be 6 for CNN")
            if self.image_shape != [224, 224, 3]:
                raise ValueError("Image shape not compatible")
        elif self.mode in {"train_mlp", "mlp_hyperparameter"}:
            if self.feature_shape != [5]:
                raise ValueError("Feature shape must be 5 for MLP")
            if self.image_shape != [224, 224, 3]:
                raise ValueError("Image shape not compatible")
        return self
    
    @model_validator(mode="after")
    def valid_params_modes(self):
        checkpoints_required = [self.checkpoint_filepath, self.monitor, self.mode]
        if self.checkpoints == "y":
            for req in checkpoints_required:
                if req == None:
                    raise ValueError("Missing parameters for checkpoints flag")
        es_required = [self.patience, self.monitor, self.mode]
        if self.early_stopping == "y":
            for req in es_required:
                if req == None:
                    raise ValueError("Missing parameters for early stopping flag")
        testing_reqs = [self.feature_scaler_path, self.model_path]
        if self.mode == "test_cnn" or self.mode == "test_mlp":
            for req in testing_reqs:
                if req == None:
                    raise ValueError("Missing parameters for testing")
        if self.mode == "test_mlp" and self.img_path == None or self.label_scaler_path == None:
            raise ValueError("Missing parameters for testing mlp. Require imgPath, label_scaler_path, feature_scaler_path, and mode_path.")
        

    
    








