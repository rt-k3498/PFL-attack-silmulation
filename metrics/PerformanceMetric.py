

class PerformanceMetric:
    def __init__(self, name: str):
        self.name = name

    def measure(self): 
        raise NotImplementedError("Subclasses must implement this method")