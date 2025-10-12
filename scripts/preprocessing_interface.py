from abc import ABC, abstractmethod
import pandas as pd

class PreprocessorInterface(ABC):
    @abstractmethod
    def preprocess_data(self, train_path, test_path)->pd.DataFrame:
        pass