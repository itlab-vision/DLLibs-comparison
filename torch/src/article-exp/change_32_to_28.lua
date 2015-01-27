require 'torch'
require 'image'

trainPath = '/home/dmitry/Projects/DNN-develop/data/MNIST/mnist.t7/train_32x32.t7'
testPath = '/home/dmitry/Projects/DNN-develop/data/MNIST/mnist.t7/test_32x32.t7'
trainData = torch.load(trainPath,'ascii')
testData = torch.load(testPath,'ascii')

trainSize = trainData.data:size(1)
testSize = testData.data:size(1)

local trdata = {
	data = torch.Tensor(60000, 1, 28, 28),
	labels = torch.Tensor(60000)
}
local tedata = {
	data = torch.Tensor(10000, 1, 28, 28),
	labels = torch.Tensor(60000)
}

for i = 1, trainSize do
	trdata.data[i] = image.scale(trainData.data[i], 28, 28)
	trdata.labels[i] = trainData.labels[i]
end
for i = 1, testSize do
	tedata.data[i] = image.scale(testData.data[i], 28, 28)
	tedata.labels[i] = testData.labels[i]
end

torch.save(trainPath, trdata, 'ascii')
torch.save(testPath, tedata, 'ascii')

print 'all is well'
