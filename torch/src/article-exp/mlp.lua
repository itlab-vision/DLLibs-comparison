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

-- define the mlp
mlp = nn.Sequential()

mlp:add(nn.Reshape(784))
mlp:add(nn.Linear(784, 392))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(392, 196))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(196, 10))

mlp:add(nn.LogSoftMax())

loss = nn.ClassNLLCriterion()

local optimState = {
   learningRate = 1e-2,
   momentum = 0.9,--0.1,
   weightDecay = 0.0005--1e-5
}

batchSize = 128

local x = torch.Tensor(batchSize,trainData.data:size(2),
         trainData.data:size(3), trainData.data:size(4))
local yt = torch.Tensor(batchSize)

-- wrap into the cuda
mlp = mlp:cuda()
loss = loss:cuda()
trainData.data = trainData.data:cuda()
testData.data = testData.data:cuda()
x = x:cuda()
yt = yt:cuda()

local w,dE_dw = mlp:getParameters()

--classes = {'1','2','3','4','5','6','7','8','9','0'}
--confusion = optim.ConfusionMatrix(classes)

local function train()
    local shuffle = torch.randperm(trainSize)

    for t = 1, trainSize, batchSize do
        --xlua.progress(t, trainSize)
        collectgarbage()

        -- batch fits?
        if (t + batchSize - 1) > trainSize then
            break
        end

        -- create mini batch
        local idx = 1
        for i = t, t + batchSize - 1 do
            x[idx] = trainData.data[shuffle[i]]
            yt[idx] = trainData.labels[shuffle[i]]
            if yt[idx] == 0 then
                yt[idx] = 1
            end
            idx = idx + 1
        end

        -- create closure to evaluate f(X) and df/dX
        local eval_E = function(w)
            -- reset gradients
            dE_dw:zero()

            -- evaluate function for complete mini batch
            local y = mlp:forward(x)
            local E = loss:forward(y,yt)

            -- estimate df/dW
            local dE_dy = loss:backward(y,yt)
            mlp:backward(x,dE_dy)

            --for i = 1,batchSize do
            --    confusion:add(y[i],yt[i])
            --end
            -- return f and df/dX
            return E, dE_dw
        end

        -- optimize on current mini-batch
        optim.sgd(eval_E, w, optimState)
    end
    --print(confusion)
    --confusion:zero()
end


local time = sys.clock()
for i = 1, 150 do
    train()
end
time = sys.clock() - time
print(time .. ' seconds needed to 150 epochs of training\n')


-- test over test data
classes = {'1','2','3','4','5','6','7','8','9','0'}
confusion = optim.ConfusionMatrix(classes)

time = sys.clock()
print('==> testing on test set:')
for t = 1,testSize do
    -- disp progress
    xlua.progress(t, testSize)

    -- get new sample
    local input = testData.data[t]
    local target = testData.labels[t]
    -- test sample
    local pred = mlp:forward(input):view(10)
    confusion:add(pred, target)
end

-- timing
time = sys.clock() - time
time = time / testSize
print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

-- print confusion matrix
print(confusion)

-- save mlp
path_to_save = "/home/dmitry/Projects/DNN-develop/torch/src/article-exp/mlp.net"
torch.save(path_to_save, mlp)
