
import pathlib

import networkx as nx

LOG_FORMAT='%(asctime)s [%(levelname)-5s] %(name)s:%(message)s'

def read_gpickle(gpickle: pathlib.Path) -> nx.Graph:
    """Read the pickle file located at path. This is just for convenience as networkx does not directly accept paths."""
    with gpickle.open('rb') as pickle_file:
        return nx.read_gpickle(pickle_file)


def read_yaml(yaml_file: pathlib.Path) -> nx.Graph:
    """Read the pickle file located at path. This is just for convenience as networkx does not directly accept paths."""
    return nx.read_yaml(str(yaml_file))


class memorize(dict):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result