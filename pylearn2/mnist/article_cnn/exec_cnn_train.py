import os
import pylearn2
from pylearn2.config import yaml_parse

path = os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials', 'convolutional_network', 'conv.yaml')
with open('conv.yaml', 'r') as f:
    train = f.read()
hyper_params = {'train_stop': 60000,
                'valid_stop': 60000,
                'test_stop': 10000,
                'save_path': '.'}
train = train % (hyper_params)

train = yaml_parse.load(train)
train.main_loop()
