import os
import pylearn2
from pylearn2.config import yaml_parse

path = os.path.join('mlp.yaml')
with open(path, 'r') as f:
     train_3 = f.read()
hyper_params = {'train_stop' : 50000,
                'valid_stop' : 60000,
                'save_path' : '.'}
train_3 = train_3 % (hyper_params)

train_3 = yaml_parse.load(train_3)
train_3.main_loop()

