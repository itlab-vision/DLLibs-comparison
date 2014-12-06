require 'torch'
require 'image'

path_to_pic = "/media/Data/The_Moon/Images/Widescreen Wallpapers Mega Pack/"
path_to_dset = "/home/dmitry/Projects/DNN-develop/data/NEG_32x32/"
names = "files.txt"

count = 0
for line in io.lines(names) do
	img = image.loadJPG(path_to_pic .. line)
	for i = 1, 10 do
		for j = 1, 6 do
			img_dst = image.crop(
				img,
				(i - 1) * 250, (j - 1) * 250,
				i * 250, j * 250
				)
			img_dst = image.scale(img_dst, 32, 32)
			str_num = string.format("%05d", count)
			image.save(path_to_dset .. str_num .. ".jpg", img_dst)
			count = count + 1
		end
	end
	print(count)
	collectgarbage()
end
