SetupEnv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if is_server
  if strcmp(dataset, 'voc12') || strcmp(dataset, 'voc10_part')
    VOC_root_folder = fullfile(WORK_ROOT_DIR, '/rmt/data/pascal/VOCdevkit');
  elseif strcmp(dataset, 'coco')
    VOC_root_folder = fullfile(WORK_ROOT_DIR, '/rmt/data/coco');
  else
    error('Wrong dataset');
  end
else
  if strcmp(dataset, 'voc12')  
    VOC_root_folder = '~/dataset/PASCAL/VOCdevkit';
  elseif strcmp(dataset, 'coco')
    VOC_root_folder = '~/dataset/coco';
  else
    error('Wrong dataset');
  end
end

if has_postprocess
  if learn_crf
    post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_ModelType%d_Epoch%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, model_type, epoch); 
  else
    post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std); 
  end
else
  post_folder = 'post_none';
end

output_mat_folders = cell(1, numel(multi_feature_type));

for i = 1 : numel(output_mat_folders)
  output_mat_folders{i} = fullfile(WORK_ROOT_DIR, '/rmt/work/deeplabel_baidu/exper', dataset, feature_name, model_name, testset, multi_feature_type{i});
end

save_folder_name = sprintf('seg_score_%dscales', numel(output_mat_folders));

save_root_folder = fullfile(WORK_ROOT_DIR, '/rmt/work/deeplabel_baidu/exper', dataset, 'res', feature_name, model_name, testset, save_folder_name, post_folder);

fprintf(1, 'Saving to %s\n', save_root_folder);

if strcmp(dataset, 'voc12')
  seg_res_dir = [save_root_folder '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  gt_dir   = fullfile(VOC_root_folder, 'VOC2012', 'SegmentationClass');
elseif strcmp(dataset, 'coco')
  seg_res_dir = [save_root_folder '/results/COCO2014/'];
  seg_root = fullfile(VOC_root_folder, '');
  gt_dir   = fullfile(VOC_root_folder, '', 'SegmentationClass');
elseif strcmp(dataset, 'voc10_part')
  seg_res_dir = [save_root_folder '/results/VOC2010_part/'];
  seg_root    = fullfile(VOC_root_folder, 'VOC2012');
  gt_dir      = fullfile(VOC_root_folder, 'VOC2012', 'SegmentationPart');
end

save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

if strcmp(dataset, 'voc12')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
elseif strcmp(dataset, 'coco')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, '');
elseif strcmp(dataset, 'voc10_part')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2010_Part');
end

if is_mat
  % crop the results
  load('pascal_seg_colormap.mat');

  output_dirs = cell(1, numel(output_mat_folders));

  for i = 1 : numel(output_dirs)
    output_dirs{i} = dir(fullfile(output_mat_folders{i}, '*.mat'));
  end

  % for each image
  for i = 1 : numel(output_dirs{1})
    if mod(i, 100) == 0
      fprintf(1, 'processing %d (%d)...\n', i, numel(output_dirs{1}));
    end

    % for each scale
    raw_results = cell(1, numel(output_mat_folders));
    for j = 1 : numel(output_dirs)
       data = load(fullfile(output_mat_folders{j}, output_dirs{j}(i).name));       
       raw_results{j} = permute(data.data, [2 1 3]);

       s = 2 ^ (j - 1);

       if s ~= 1
         raw_results{j} = imresize(raw_results{j}, s, 'bilinear');
       end
    end

    img_fn = output_dirs{1}(i).name(1:end-4);
    img_fn = strrep(img_fn, '_blob_0', '');
    
    if strcmp(dataset, 'voc12') || strcmp(dataset, 'voc10_part')
      img = imread(fullfile(VOC_root_folder, 'VOC2012', 'JPEGImages', [img_fn, '.jpg']));
    elseif strcmp(dataset, 'coco')
      img = imread(fullfile(VOC_root_folder, 'JPEGImages', [img_fn, '.jpg']));
    end
    
    img_row = size(img, 1);
    img_col = size(img, 2);

    all_score = zeros(img_row, img_col, size(raw_results{j}, 3));    
    for j = 1 : numel(output_dirs)
      all_score = all_score + raw_results{j}(1:img_row, 1:img_col, :);
    end
    
    if ~is_argmax
      [~, result] = max(all_score, [], 3);
      result = uint8(result) - 1;
    else
      result = uint8(result);
    end

    if debug
        gt = imread(fullfile(gt_dir, [img_fn, '.png']));
        figure(1), 
        subplot(221),imshow(img), title('img');
        subplot(222),imshow(gt, colormap), title('gt');
        subplot(224), imshow(result,colormap), title('predict');
        pause();
    end
    
    imwrite(result, colormap, fullfile(save_result_folder, [img_fn, '.png']));
  end
end

% get iou score
if strcmp(testset, 'val') || strcmp(testset, 'train')
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);
else
  fprintf(1, 'This is test set. No evaluation. Just saved as png\n');
end 

    
    

