gpu_id = 0;  % -1 means use cpu only
dataset = 'voc10_part';

WORK_ROOT_DIR = '/home/lcchen/workspace/rmt';
CAFFE_ROOT    = 'work/deeplabel_baidu/code';
IMG_PATH = fullfile(WORK_ROOT_DIR, 'data/pascal/VOCdevkit/VOC2012/JPEGImages');
WORK_DIR = fullfile('work/deeplabel_baidu/exper', dataset);

net_name = 'deconv_exp18';
img_name = '2008_000579';
k = 16;
img_dim = 32 * k - 31;  % 321

if exist(fullfile(WORK_ROOT_DIR, CAFFE_ROOT, 'matlab/+caffe'), 'dir')
  addpath(fullfile(WORK_ROOT_DIR, CAFFE_ROOT, 'matlab'));
else
  error('Cannot find CAFFE')
end

% Set caffe mode
if gpu_id > -1
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

config = fullfile(WORK_ROOT_DIR, WORK_DIR, 'config', net_name, 'deploy.prototxt');
model  = fullfile(WORK_ROOT_DIR, WORK_DIR, 'model', net_name, 'train_iter_70000.caffemodel');
phase = 'test';
net = caffe.Net(config, model, phase);

im = imread(fullfile(IMG_PATH, [img_name, '.jpg']));

im = imresize(im, [img_dim img_dim], 'bilinear');
% prepare oversampled input
% input_data is Height x Width x Channel x Num
tic;
input_data = {PrepareImageForMatCaffe(im)};
toc;

% do forward pass to get scores
% scores are now Channels x Num, where Channels == 1000
tic;
% The net forward function. It takes in a cell array of N-D arrays
% (where N == 4 here) containing data of input blob(s) and outputs a cell
% array containing data from output blob(s)
scores = net.forward(input_data);
toc;

scores = scores{1};

scores = permute(scores, [2 1 3]);
[~, maxlabel] = max(scores, [], 3);
load('pascal_seg_colormap.mat')
figure(1), imshow(maxlabel, colormap), title(sprintf('img size %d', img_dim))


% call caffe.reset_all() to reset caffe
caffe.reset_all();
