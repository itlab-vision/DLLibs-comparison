import os
import pylearn2
from pylearn2.config import yaml_parse

#1. load dataset
dirname = os.path.abspath(os.path.dirname('softmax_regression.ipynb'))
with open(os.path.join(dirname, 'sr_dataset.yaml'), 'r') as f:
    dataset = f.read()
hyper_params = {'train_stop' : 50000}
dataset = dataset % (hyper_params)

#2. read model information
with open(os.path.join(dirname, 'sr_model.yaml'), 'r') as f:
    model = f.read()

#3. read training algorithm information
with open(os.path.join(dirname, 'sr_algorithm.yaml'), 'r') as f:
    algorithm = f.read()
hyper_params = {'batch_size' : 10000,
                'valid_stop' : 60000}
algorithm = algorithm % (hyper_params)

#4. load full training information
with open(os.path.join(dirname, 'sr_train.yaml'), 'r') as f:
    train = f.read()
save_path = '.'
train = train %locals()

#5. train model
train = yaml_parse.load(train)
train.main_loop()
