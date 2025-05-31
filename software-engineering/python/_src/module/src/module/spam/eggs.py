def eggs_function():
    return "This is eggs function in spam package"

class EggsClass:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello from {self.name} in spam.eggs"
