import random


class Registry(object):
    def __init__(self, name):
        self.name = name
        self.module_dict = dict()

    def get(self, query):
        key, kwargs = query
        return self.module_dict.get(key, None)(**kwargs)

    def register_module(self, cls):
        self.module_dict[cls.__name__] = cls
        return cls


class Probabilistic(Registry):
    def get(self, query):
        key, kwargs, prob = query
        cls = self.module_dict.get(key, None)(**kwargs)
        func = cls.forward

        def wrapper(images):
            if prob < random.random():
                return images
            return func(images)

        cls.forward = wrapper
        return cls


class Registry_Assembly(object):
    def __init__(self, name):
        self.name = name
        self.DIRECT = Registry(name)
        self.PROBABILISTIC = Probabilistic(name)

    def register_module(self, cls):
        for registry in (self.DIRECT, self.PROBABILISTIC):
            registry.module_dict[cls.__name__] = cls
        return cls

    def get(self, query):
        if len(query) == 2:
            return self.DIRECT.get(query)
        return self.PROBABILISTIC.get(query)


