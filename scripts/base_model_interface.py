from abc import ABC, abstractmethod

import pickle

class ModelInterface(ABC):
    @abstractmethod
    def load_data(self, features_path, labels_path):
        pass
    
    @abstractmethod
    def train_model(self, X_train, y_train):
        pass
    
    @abstractmethod
    def save_model(self, model, model_path)->pickle.dump:
        pass