SetupEnv;

addpath('/rmt/work/deeplabel/code_new/matlab');   % use matcaffe

%%% env setup
load('pascal_seg_colormap.mat');

if strcmp(dataset, 'voc12')
  VOC_root_folder = '/rmt/data/pascal/VOCdevkit';
elseif strcmp(dataset, 'coco')
  VOC_root_folder = '/rmt/data/coco';
else
  error('Wrong dataset');
end

post_folder = 'post_none';

save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_name, model_name, testset, feature_type, post_folder);

fprintf(1, 'Saving results to %s\n', save_root_folder);

if strcmp(dataset, 'voc12')
  seg_res_dir = [save_root_folder '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  % gt_dir   = fullfile(VOC_root_folder, 'VOC2012', 'SegmentationClass');
elseif strcmp(dataset, 'coco')
  seg_res_dir = [save_root_folder '/results/COCO2014/'];
  seg_root = fullfile(VOC_root_folder, '');
  % gt_dir   = fullfile(VOC_root_folder, '', 'SegmentationClass');
end

save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);
if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

save_feature_folder = fullfile('/rmt/work/deeplabel/exper', dataset, feature_name, model_name, testset, feature_type);

fprintf(1, 'Saving features to %s\n', save_feature_folder);

if ~exist(save_feature_folder, 'dir')
    mkdir(save_feature_folder);
end

if strcmp(dataset, 'voc12')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
elseif strcmp(dataset, 'coco')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, '');
end

%%% set up network
model_weight = fullfile('/rmt/work/deeplabel/exper', dataset, 'model', model_name, weight_file_name);

model_structure = fullfile('/rmt/work/deeplabel/exper', dataset, 'config', model_name, deploy_file_name);
network = caffe.Net(model_structure, model_weight, 'test');
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

%%% run inference for each image
mean_pixels = [104.008, 116.669, 122.675];
mean_pixels = reshape(mean_pixels, [1 1 3]);

num_class = VOCopts.nclasses + 1;

% get image list for inference
[gtids, t] = textread(sprintf(VOCopts.seg.imgsetpath, VOCopts.testset), '%s %d');

tic;
for k = 1 : length(gtids)
  % display progress
  if toc > 1
    fprintf('processing: %d/%d\n', k, length(gtids));
    drawnow;
    tic;
  end

  imname = gtids{k};

  % read image
  img = imread(sprintf(VOCopts.imgpath, imname));
  img = single(img(:, :, [3 2 1]));  % change to BRG order
  img = permute(img, [2 1 3]);       % caffe is row-major

  % subtract the mean
  img_minus_mean = bsxfun(@minus, img, mean_pixels);

  [img_row, img_col, img_channel] = size(img_minus_mean);

  % save results for each scale
  scale_results = cell(1, numel(input_scales));

  for m = 1 : numel(input_scales)
    [img_pad, res_row, res_col] = PadAndScaleImage(img_minus_mean, input_scales(m), input_max_size);
    res = network.forward({single(img_pad)});
  
 %%   % in the prob space
 %%   tmp = res{1}(1 : res_row, 1 : res_col, :);
 %%   tmp = imresize(tmp, [img_row, img_col]);
 %%   tmp = bsxfun(@times, exp(tmp), 1./sum(exp(tmp),3));

    % in the score space
    tmp = res{1}(1 : res_row, 1 : res_col, :);
    tmp = imresize(tmp, [img_row, img_col]);

    scale_results{m} = tmp;
  end

%%  % noisy-or for foreground classes and complement for background class
%%  output = zeros(img_row, img_col, num_class);
%%  tmp = 1;
%%  for s = 1 : numel(input_scales)
%%    tmp = tmp .* ( 1 - scale_results{s}(:, :, 2:end));
%%  end
%%  output(:, :, 2:end) = 1 - tmp;
%%  output(:, :, 1) = 1 - sum(tmp, 3);    % 1 - max(tmp, [], 3);

  % compute avg
  tmp = 0;
  for s = 1 : numel(input_scales)
    tmp = tmp + scale_results{s};
  end
  output = tmp / numel(input_scales);

  % save features in score space for densecrf
  data = output;
  save(fullfile(save_feature_folder, [imname, '_blob_0.mat']), 'data');

  % save map result
  [~, label] = max(output, [], 3);
  imwrite(label', colormap, fullfile(save_result_folder, [imname, '.png']));
end
  
caffe.reset_all();
