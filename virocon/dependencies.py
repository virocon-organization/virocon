from inspect import signature


class DependenceFunction():
    
    #TODO implement check of bounds and constraints
    def __init__(self, func, bounds=None, constraints=None):
        #TODO add fitting method 
        self.func = func
        self.bounds = bounds
        self.constraints = constraints
        
        # read default values from function or set default as 1 if not specified
        sig = signature(func)
        self.parameters = {par.name : (par.default if par.default is not par.empty else 1) 
                           for par in list(sig.parameters.values())[1:]
                           }
        
        
    def __call__(self, x, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            return self.func(x, *self.parameters.values())
        elif len(args) + len(kwargs) == len(self.parameters):
            return self.func(x, *args, **kwargs)
        else:
            raise ValueError() # TODO helpful error message