----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'gfx.js'  -- to visualize the dataset
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- training/test size

trsize = 60000
tesize = 10000

----------------------------------------------------------------------
print '==> loading dataset'

loaded = torch.load('/home/dmitry/data/mnist/train.th7','binary')
trainData = {
   data = loaded[1]:transpose(2, 4):transpose(3, 4),
   labels = loaded[2],
   size = function() return trsize end
}

loaded = torch.load('/home/dmitry/data/mnist/test.th7','binary')
testData = {
   data = loaded[1]:transpose(2, 4):transpose(3, 4),
   labels = loaded[2],
   size = function() return tesize end
}

----------------------------------------------------------------------
print '==> preprocessing data'

mean = trainData.data[{}]:mean()
std = trainData.data[{}]:std()
trainData.data[{}]:add(-mean)
trainData.data[{}]:div(std)


-- Normalize test data, using the training means/stds
-- normalize each channel globally:
testData.data[{}]:add(-mean)
testData.data[{}]:div(std)

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

trainMean = trainData.data[{}]:mean()
trainStd = trainData.data[{}]:std()

testMean = testData.data[{}]:mean()
testStd = testData.data[{}]:std()

print('training data, mean: ' .. trainMean)
print('training data, standard deviation: ' .. trainStd)

print('test data, mean: ' .. testMean)
print('test data, standard deviation: ' .. testStd)


for i = 1, trainData:size() do
	if trainData.labels[i] == 0 then
		trainData.labels[i] = 10
	end
end

for i = 1, testData:size() do
	if testData.labels[i] == 0 then
		testData.labels[i] = 10
	end
end

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using gfx.image().

if opt.visualize then
   gfx.image(trainData.data[{ {1,25} }], {legend=''})
end
