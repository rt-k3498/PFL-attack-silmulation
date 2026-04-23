from typing import Literal, Dict, Any
import tensorflow as tf 

from models.model import Model

SettingOptions = Literal["layers"]
Settings = Dict[SettingOptions, Any]

class CNN(Model):

    def __init__(self, settings: Settings = {}):
        super().__init__(settings)
        

