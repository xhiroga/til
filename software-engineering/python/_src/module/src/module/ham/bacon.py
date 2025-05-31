def bacon_function():
    return "This is bacon function in ham namespace package"

class BaconClass:
    def __init__(self, flavor):
        self.flavor = flavor
    
    def describe(self):
        return f"This is {self.flavor} bacon in ham namespace package"