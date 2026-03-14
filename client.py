from typing import List
from models.model import Model
from data.data import Data

class Client:
    def __init__(self, id: int, model: Model, data: Data):
        self.id = id
        self.model = model
        self.data = data