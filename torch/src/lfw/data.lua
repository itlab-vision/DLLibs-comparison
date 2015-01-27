require 'torch'   -- torch
require 'cutorch'
require 'image'   -- to visualize the dataset
require 'nn'      -- provides a normalization operator

classes = { 'face', 'backg' }

path_to_ready_dset = "/home/dmitry/Projects/DNN-develop/data/LFW_Torch/"
path_to_neg = "/home/dmitry/Projects/DNN-develop/data/pascal_voc/"
path_to_pos = "/home/dmitry/Projects/DNN-develop/data/LFW/"

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> loading dataset')

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end


if (
   file_exists(path_to_ready_dset .. "train.th7") and
   file_exists(path_to_ready_dset .. "test.th7")) then

   print "loading dataset from disk"
   trainData = torch.load(path_to_ready_dset .. "train.th7")
   testData = torch.load(path_to_ready_dset .. "test.th7")

   print(sys.COLORS.red .. "==> verify datasets")
   print "train"
   print(trainData)
   print "test"
   print(testData)

else
   print "now we will create dataset"

   if not file_exists(path_to_neg .. "annotation.txt") then
      print "Please create annotation for negative dataset!"
      return
   end

   neg_annot = io.open(path_to_neg .. "annotation.txt", "r")

   neg_size = neg_annot:read("*n")
   neg_annot:read() -- read \n after digit

   if neg_size == nil then
      print "Please write in annotation the size of dataset (first string)"
      return
   end

   pos_size = 13234

   sum_size = pos_size + neg_size

   local imagesAll = torch.Tensor(sum_size, 3, 32, 32)
   local labelsAll = torch.Tensor(sum_size)

   -- load negatives:
   for i = 0, neg_size do
      path = neg_annot:read()
      imagesAll[i + 1] = image.loadJPG(path_to_neg .. path)
      labelsAll[i + 1] = 2
   end
   collectgarbage()
   print 'negative dataset has been loaded'

   -- load faces:
   lines = io.lines(path_to_pos .. "name_folders.txt")

   i = neg_size + 1
   for line in lines do
      if string.find(line, "0") == nil then
         cur_path = path_to_pos .. line .. "/"
         collectgarbage()
      else
         imagesAll[i] = image.scale(
                        image.loadJPG(cur_path .. line),
                        32, 32)
         labelsAll[i] = 1
         i = i + 1
      end
   end
   collectgarbage()
   print 'positive dataset has been loaded'

   -- shuffle dataset: get shuffled indices in this variable:
   local labelsShuffle = torch.randperm((#labelsAll)[1])

   local portionTrain = 0.8 -- 80% is train data, rest is test data
   local trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
   local tesize = labelsShuffle:size(1) - trsize

   trainData = {
      data = torch.Tensor(trsize, 3, 32, 32),
      labels = torch.Tensor(trsize),
      size = function() return trsize end
   }

   testData = {
      data = torch.Tensor(tesize, 3, 32, 32),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }

   for i = 1, trsize do
      trainData.data[i] = imagesAll[labelsShuffle[i]]:clone()
      trainData.labels[i] = labelsAll[labelsShuffle[i]]
   end
   collectgarbage()
   for i = trsize + 1, tesize + trsize do
      testData.data[i-trsize] = imagesAll[labelsShuffle[i]]:clone()
      testData.labels[i-trsize] = labelsAll[labelsShuffle[i]]
   end

   -- remove from memory temp image files:
   --imagesAll = nil
   --labelsAll = nil
   collectgarbage()


   ----------------------------------------------------------------------

   print(sys.COLORS.red ..  '==> preprocessing data')
   -- Preprocessing requires a floating point representation (the original
   -- data is stored on bytes). Types can be easily converted in Torch,
   -- in general by doing: dst = src:type('torch.TypeTensor'),
   -- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
   -- for simplicity (float(),double(),cuda(),...):

   trainData.data = trainData.data:float()
   testData.data = testData.data:float()

   -- We now preprocess the data. Preprocessing is crucial
   -- when applying pretty much any kind of machine learning algorithm.

   -- For natural images, we use several intuitive tricks:
   --   + images are mapped into YUV space, to separate luminance information
   --     from color information
   --   + the luminance channel (Y) is locally normalized, using a contrastive
   --     normalization operator: for each neighborhood, defined by a Gaussian
   --     kernel, the mean is suppressed, and the standard deviation is normalized
   --     to one.
   --   + color channels are normalized globally, across the entire dataset;
   --     as a result, each color component has 0-mean and 1-norm across the dataset.

   --Convert all images to YUV
   print '==> preprocessing data: colorspace RGB -> YUV'
   for i = 1,trainData:size() do
      trainData.data[i] = image.rgb2yuv(trainData.data[i])
   end
   for i = 1,testData:size() do
      testData.data[i] = image.rgb2yuv(testData.data[i])
   end

   -- Name channels for convenience
   local channels = {'y','u','v'}

   -- Normalize each channel, and store mean/std
   -- per channel. These values are important, as they are part of
   -- the trainable parameters. At test time, test data will be normalized
   -- using these values.
   print(sys.COLORS.red ..  '==> preprocessing data: normalize each feature (channel) globally')

   file_stat = io.open(path_to_ready_dset .. "statistics.txt", "w")

   local mean = {}
   local std = {}
   file_stat:write("before\n")
   for i,channel in ipairs(channels) do
      -- normalize each channel globally:
      mean[i] = trainData.data[{ {},i,{},{} }]:mean()
      std[i] = trainData.data[{ {},i,{},{} }]:std()
      print('mean channel '..i..' = '..mean[i])
      print('std channel '..i..' = '..std[i])
      file_stat:write('mean channel '..i..' = '..mean[i] .. "\n")
      file_stat:write('std channel '..i..' = '..std[i] .. "]\n")
      trainData.data[{ {},i,{},{} }]:add(-mean[i])
      trainData.data[{ {},i,{},{} }]:div(std[i])
   end
   file_stat:write("\n")

   -- Normalize test data, using the training means/stds
   for i,channel in ipairs(channels) do
      -- normalize each channel globally:
      testData.data[{ {},i,{},{} }]:add(-mean[i])
      testData.data[{ {},i,{},{} }]:div(std[i])
   end

   -- Local contrast normalization is needed in the face dataset as the dataset is already in this form:
   print(sys.COLORS.red ..  '==> preprocessing data: normalize all three channels locally')

   -- Define the normalization neighborhood:
   local neighborhood = image.gaussian1D(5) -- 5 for face detector training

   -- Define our local normalization operator (It is an actual nn module,
   -- which could be inserted into a trainable model):
   local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

   -- Normalize all channels locally:
   for c in ipairs(channels) do
      for i = 1,trainData:size() do
         trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
      end
      for i = 1,testData:size() do
         testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
      end
   end

   ----------------------------------------------------------------------
   print(sys.COLORS.red ..  '==> verify statistics')

   file_stat:write("after\n")
   for i,channel in ipairs(channels) do
      local trainMean = trainData.data[{ {},i }]:mean()
      local trainStd = trainData.data[{ {},i }]:std()

      local testMean = testData.data[{ {},i }]:mean()
      local testStd = testData.data[{ {},i }]:std()

      print('training data, '..channel..'-channel, mean: ' .. trainMean)
      print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)
      file_stat:write('training data, '..channel..'-channel, mean: ' .. trainMean .. "\n")
      file_stat:write('training data, '..channel..'-channel, standard deviation: ' .. trainStd .. "\n")

      print('test data, '..channel..'-channel, mean: ' .. testMean)
      print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
      file_stat:write('test data, '..channel..'-channel, mean: ' .. testMean .. "\n")
      file_stat:write('test data, '..channel..'-channel, standard deviation: ' .. testStd .. "\n")
   end
   file_stat:close()

   ----------------------------------------------------------------------
   print(sys.COLORS.red ..  '==> preprocessing data is done, saving...')
   torch.save(path_to_ready_dset .. 'train.th7', trainData)
   torch.save(path_to_ready_dset .. 'test.th7', testData)
end

-- Exports
return {
   trainData = trainData,
   testData = testData,
   classes = classes
}
