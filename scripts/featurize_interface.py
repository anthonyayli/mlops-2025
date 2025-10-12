from abc import ABC, abstractmethod
import pandas as pd


class FeaturizerInterface(ABC):
    @abstractmethod
    def engineering_data(self, df)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass
    
    @abstractmethod
    def get_cat_tranformation(self):
        pass
    
    @abstractmethod
    def get_bins(self):
        pass