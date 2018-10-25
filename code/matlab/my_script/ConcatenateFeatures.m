% concatenate the features for training svm
%

SetupEnv;

num_class = 20; % hard coded

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gt_folder = '/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClassAug';

output_mat_folder = fullfile('/rmt/work/deeplabel/exper', dataset, feature_name, model_name, testset, feature_type);

save_mat_folder = fullfile(output_mat_folder, 'concat');

fprintf(1, 'Saving to %s\n', save_mat_folder);

if ~exist(save_mat_folder, 'dir')
    mkdir(save_mat_folder);
end

output_dir = dir(fullfile(output_mat_folder, '*.mat'));

% read one mat file to get feature dimension
tmp = load(fullfile(output_mat_folder, output_dir(1).name));

feat_dim = numel(tmp.data);
[num_bin_x, num_bin_y, one_bin_dim] = size(tmp.data);

num_img  = numel(output_dir);
feat     = zeros(feat_dim, num_img);

labels   = -1 * ones(num_class, num_img);

fprintf(1, 'feat dim: %d, num img: %d\n', feat_dim, num_img);

for i = 1 : numel(output_dir)
  if mod(i, 100) == 0
      fprintf(1, 'processing %d (%d)...\n', i, numel(output_dir));
  end

  % get features
  tmp = load(fullfile(output_mat_folder, output_dir(i).name));
  tmp = (reshape(tmp.data, num_bin_x*num_bin_y, one_bin_dim))';
  feat(:, i) = tmp(:);  
  
  % get label 
  img_fn = output_dir(i).name(1:end-4);
  img_fn = strrep(img_fn, '_blob_0', '');

  gt = imread(fullfile(gt_folder, [img_fn, '.png']));
  gt = unique(gt(:));
  ind = gt ~= 0 & gt ~= 255;
  gt = gt(ind);

  for k = 1 : length(gt)
    label = gt(k);
  
    assert(label >= 1 && label <= num_class);

    labels(label, i) = 1;
  end 
end

save(fullfile(save_mat_folder, 'feat_matrix_and_label.mat'), 'feat', 'labels');
