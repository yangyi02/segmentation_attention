function crops_data = PrepareImageForMatCaffe(im)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels

% in BGR order
mean_pixel = reshape([104.008 116.669 122.675], [1 1 3]);

% change image to BGR order
im_data = im(:, :, [3 2 1]);
im_data = permute(im_data, [2 1 3]);  % flip width and height
im_data = single(im_data);
im_data = bsxfun(@minus, im_data, mean_pixel);

crops_data = zeros(size(im_data, 1), size(im_data, 2), 3, 1, 'single');
crops_data(:, :, :, 1) = im_data;