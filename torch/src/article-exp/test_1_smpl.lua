require 'torch'
require 'nn'
require 'optim'
require 'xlua'

require 'cutorch'
require 'cunn'

trainPath = '/home/dmitry/Projects/DNN-develop/data/MNIST/mnist.t7/train_32x32.t7'
testPath = '/home/dmitry/Projects/DNN-develop/data/MNIST/mnist.t7/test_32x32.t7'
trainData = torch.load(trainPath,'ascii')
testData = torch.load(testPath,'ascii')

trainSize = 50000
testSize = 10000

-- normalize data
std = trainData.data[{ {1, trainSize} }]:std()
trainData.data[{ {1, trainSize} }]:div(std)
testData.data[{ {1, testSize} }]:div(std)

--trainData.data = trainData.data:cuda()
--testData.data = testData.data:cuda()

model = torch.load('cnn.net'):double()

print('==> testing on test set:')

size = 10000

time = sys.clock()
for t = 1, size do
    local pred = model:forward(testData.data[t]):view(10)
end
time = sys.clock() - time

-- timing
time = time / size
print("\n==> time to test 1 sample = " .. (time * 1000) .. 'ms')
