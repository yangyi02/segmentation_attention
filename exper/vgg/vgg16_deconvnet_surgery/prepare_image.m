function [out, resize_out] = prepare_image(im)
% prepare the input im for cnn
%
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
mean_pixel = reshape(single([104.008, 116.669, 122.675]), [1 1 3]);

IMAGE_DIM = 224;
% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
im_data = bsxfun(@minus, im_data, mean_pixel);  % subtract mean_pixel

resize_out = imresize(im, [IMAGE_DIM, IMAGE_DIM], 'bilinear');

out = zeros(IMAGE_DIM, IMAGE_DIM, 3, 1, 'single');
out(:, :, :, 1) = im_data;
