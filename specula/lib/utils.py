
import re
import typing
import importlib


def camelcase_to_snakecase(s):
    tokens = re.findall('[A-Z]+[0-9a-z]*', s)
    return '_'.join([x.lower() for x in tokens])


def import_class(classname):
    modulename = camelcase_to_snakecase(classname)
    try:
        try:
            mod = importlib.import_module(f'specula.processing_objects.{modulename}')
        except ModuleNotFoundError:
            try:
                mod = importlib.import_module(f'specula.data_objects.{modulename}')
            except ModuleNotFoundError:
                mod = importlib.import_module(f'specula.display.{modulename}')
    except ModuleNotFoundError:
        raise ImportError(f'Class {classname} must be defined in a file called {modulename}.py but it cannot be found')
    
    try:
        return getattr(mod, classname)
    except AttributeError:
        raise AttributeError(f'Class {classname} not found in file {modulename}.py')


def get_type_hints(type):
    hints ={}
    for x in type.__mro__:
        hints.update(typing.get_type_hints(getattr(x, '__init__')))
    return hints
