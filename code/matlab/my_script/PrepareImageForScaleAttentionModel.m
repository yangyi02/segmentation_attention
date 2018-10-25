function crops_data = PrepareImageForScaleAttentionModel(im, input_dim)
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

[img_row, img_col, img_channel] = size(im_data);
pad_img = zeros(input_dim, input_dim, 3);
for c = 1 : img_channel
    pad_img(1:img_row, 1:img_col, c) = im_data(:, :, c);
end

crops_data = zeros(input_dim, input_dim, img_channel, 1, 'single');
crops_data(:, :, :, 1) = pad_img;