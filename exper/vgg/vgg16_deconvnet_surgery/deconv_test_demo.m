%%% variables
use_gpu = 1;
gpu_id  = 0;

orig_prototxt = './init0.prototxt';
orig_weights  = './init0.caffemodel';

%%%

% Add caffe/matlab to you Matlab search PATH to use matcaffe
if exist('../../../code/matlab/+caffe', 'dir')
  addpath('../../../code/matlab');
else
  error('Please run this demo from caffe/matlab/demo');
end

% Set caffe mode
if use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

phase = 'test';

net_model = orig_prototxt;
net_weights = orig_weights;
net = caffe.Net(net_model, net_weights, phase);

% prepare data
%im = imread('./data/2007_003194.jpg');
im = imread('./data/2007_003329.jpg');
%im = imread('./data/2007_000129.jpg');

[im_data, resize_data] = prepare_image(im);
im_data = {im_data};

% forward pass
scores = net.forward(im_data);

class_id = 16;  % 13: dog, 16: person
tmp = exp(scores{1}(:, :, class_id)) ./ sum(exp(scores{1}), 3);
tmp = tmp';

[~, label] = max(scores{1}, [], 3);
mask = label == class_id;
mask = mask';
masked_im = zeros(size(resize_data), 'uint8');
for c = 1 : 3
    tt = zeros(size(mask));
    re_data = resize_data(:, :, c);
    tt(mask) = re_data(mask);
    tt(~mask) = 0.4 * re_data(~mask);
    masked_im(:, :, c) = uint8(tt);
end
figure(1), subplot(221), imshow(im), subplot(222), imagesc(tmp), subplot(224), imshow(masked_im);

caffe.reset_all();
