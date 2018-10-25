clear all; close all;

gpu_id = 1;  % -1 means use cpu only
dataset = 'voc10_part';

WORK_ROOT_DIR = '/home/lcchen/workspace/rmt';
CAFFE_ROOT    = 'work/deeplabel_baidu/code';
IMG_PATH = fullfile(WORK_ROOT_DIR, 'data/pascal/VOCdevkit/VOC2012/JPEGImages');
WORK_DIR = fullfile('work/deeplabel_baidu/exper', dataset);
tmp=load('pascal_seg_colormap.mat');

net_name = 'vgg128_noup_pool3_20M_largewin_attention17';
img_name = '2008_000579';   %'2008_000579';
input_dim = 513;

%if exist(fullfile(WORK_ROOT_DIR, CAFFE_ROOT, 'matlab/+caffe'), 'dir')
%  addpath(fullfile(WORK_ROOT_DIR, CAFFE_ROOT, 'matlab'));
%else
%  error('Cannot find CAFFE')
%end
addpath('../../matlab');

% Set caffe mode
if gpu_id >= 0
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

im = imread('test_img.png');
im = imresize(im, [513, 513]);
%im = imread(fullfile(IMG_PATH, [img_name, '.jpg']));
[img_row, img_col, img_channels] = size(im);

config = 'models/deploy.prototxt';
model = 'models/train_iter_6000.caffemodel';
%config = fullfile(WORK_ROOT_DIR, WORK_DIR, 'config', net_name, 'deploy.prototxt');
%model  = fullfile(WORK_ROOT_DIR, WORK_DIR, 'model', net_name, 'train_iter_6000.caffemodel');
phase = 'test';
net = caffe.Net(config, model, phase);
% prepare oversampled input
% input_data is Height x Width x Channel x Num
tic;
input_data = {PrepareImageForScaleAttentionModel(im, input_dim)};
toc;

% do forward pass to get scores
% scores are now Channels x Num, where Channels == 1000
tic;
% The net forward function. It takes in a cell array of N-D arrays
% (where N == 4 here) containing data of input blob(s) and outputs a cell
% array containing data from output blob(s)
scores = net.forward(input_data);
toc;

% show predicted labels
scores = scores{1};
scores = permute(scores, [2 1 3]);
scores = scores(1:img_row, 1:img_col, :);
[~, maxlabel] = max(scores, [], 3);
maxlabel = maxlabel;

% show learned attention
attention = net.blobs('attention').get_data();
attention = permute(attention, [2 1 3]);
attention = imresize(attention, [input_dim, input_dim], 'bilinear');
attention = attention(1:img_row, 1:img_col, :);
num_att = size(attention, 3);

figure(1)
subplot(2, num_att, 1), imshow(im), title(sprintf('image %s', img_name))
subplot(2, num_att, 2), imshow(uint8(maxlabel), tmp.colormap), title('part seg')
freezeColors

for c = 1 : num_att
    subplot(2, num_att, c + num_att)
    colormap(jet)
    imagesc(squeeze(attention(:,:,c)))
    colorbar, axis square
    title(sprintf('attention %d', c))
end

% call caffe.reset_all() to reset caffe
caffe.reset_all();
