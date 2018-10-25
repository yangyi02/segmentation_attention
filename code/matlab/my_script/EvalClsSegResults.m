SetupEnv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if is_server
  if strcmp(dataset, 'voc12')
    VOC_root_folder = '/rmt/data/pascal/VOCdevkit';
  elseif strcmp(dataset, 'coco')
    VOC_root_folder = '/rmt/data/coco';
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

output_mat_folder = fullfile('/rmt/work/deeplabel/exper', dataset, feature_name, model_name, testset, feature_type);

save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_name, model_name, testset, feature_type, post_folder);

fprintf(1, 'Saving to %s\n', save_root_folder);

if strcmp(dataset, 'voc12')
  seg_res_dir = [save_root_folder sprintf('/results_%s/VOC2012/', cls_score_type)];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  seg_gt_dir   = fullfile(VOC_root_folder, 'VOC2012', seg_gt_task_folder);

  all_res_dir = [save_root_folder, sprintf('/my_results_%s/VOC2012/', cls_score_type)];
elseif strcmp(dataset, 'coco')
  seg_res_dir = [save_root_folder sprintf('/results_%s/COCO2014/', cls_score_type)];
  seg_root = fullfile(VOC_root_folder, '');
  seg_gt_dir   = fullfile(VOC_root_folder, '', seg_gt_task_folder);

  all_res_dir = [save_root_folder, sprintf('/my_results_%s/COCO2014/', cls_score_type)];
end

% save all the results (since cls and seg have non-overlapped sets)
save_seg_result_folder = fullfile(all_res_dir, seg_task_folder, [seg_id '_' testset '_cls']);
save_cls_result_folder = fullfile(all_res_dir, cls_task_folder);

% only save the results that will be evaulated
eval_seg_result_folder = fullfile(seg_res_dir, seg_task_folder, [seg_id '_' testset '_cls']);
eval_cls_result_folder = fullfile(seg_res_dir, cls_task_folder);


if ~exist(save_seg_result_folder, 'dir')
    mkdir(save_seg_result_folder);
end

if ~exist(save_cls_result_folder, 'dir')
    mkdir(save_cls_result_folder);
end

if ~exist(eval_seg_result_folder, 'dir')
    mkdir(eval_seg_result_folder);
end

if ~exist(eval_cls_result_folder, 'dir')
    mkdir(eval_cls_result_folder);
end


if strcmp(dataset, 'voc12')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
elseif strcmp(dataset, 'coco')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, '');
end

if is_mat
  % crop the results
  load('pascal_seg_colormap.mat');

  output_dir = dir(fullfile(output_mat_folder, '*.mat'));

  % iterate through all images (note cls and seg have non-overlapped sets)
  for i = 1 : numel(output_dir)
    if mod(i, 100) == 0
        fprintf(1, 'processing %d (%d)...\n', i, numel(output_dir));
    end

    if strcmp(feature_type, 'fc8_spm')
      % the result is ONLY saved in the format for classification

      data = load(fullfile(output_mat_folder, output_dir(i).name));
      raw_result = squeeze(data.data);

      img_fn = output_dir(i).name(1:end-4);
      img_fn = strrep(img_fn, '_blob_0', '');

      for kk = 1 : numel(VOCopts.classes)
         cls_res_fn = sprintf('%s_cls_%s_%s.txt', cls_id, testset, VOCopts.classes{kk});

         if i == 1
           cls_fid = fopen(fullfile(save_cls_result_folder, cls_res_fn), 'w');
         else
           cls_fid = fopen(fullfile(save_cls_result_folder, cls_res_fn), 'a');
         end

         fprintf(cls_fid, '%s %f\n', img_fn, raw_result(kk+1));

         fclose(cls_fid);
      end

    else
      % get segmentation results
      data = load(fullfile(output_mat_folder, output_dir(i).name));
      raw_result = data.data;
      raw_result = permute(raw_result, [2 1 3]);

      img_fn = output_dir(i).name(1:end-4);
      img_fn = strrep(img_fn, '_blob_0', '');
    
      if strcmp(dataset, 'voc12')
        img = imread(fullfile(VOC_root_folder, 'VOC2012', 'JPEGImages', [img_fn, '.jpg']));
      elseif strcmp(dataset, 'coco')
        img = imread(fullfile(VOC_root_folder, 'JPEGImages', [img_fn, '.jpg']));
      end
    
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      raw_result = raw_result(1:img_row, 1:img_col, :);

      if ~is_argmax
        [tmp_raw_result, result] = max(raw_result, [], 3);
        result = uint8(result) - 1;
      else
        result = uint8(raw_result);
      end

      if debug
          gt = imread(fullfile(seg_gt_dir, [img_fn, '.png']));
          figure(1), 
          subplot(221),imshow(img), title('img');
          subplot(222),imshow(gt, colormap), title('gt');
          subplot(224), imshow(result,colormap), title('predict');
      end
    
      imwrite(result, colormap, fullfile(save_seg_result_folder, [img_fn, '.png']));

      % get classification results
      % save results with file name like "comp2_cls_test_person.txt"

      if strcmp(cls_score_type, 'soft')
        %Transform data to probability
        tmp_raw_result = bsxfun(@minus, raw_result, tmp_raw_result);

        pred_prob = exp(tmp_raw_result);
        pred_prob = bsxfun(@rdivide, pred_prob, sum(pred_prob, 3));
      end

      for kk = 1 : numel(VOCopts.classes)
         if strcmp(cls_score_type, 'hard')
           pred = result == kk;
           pred = sum(pred(:)) / (img_row * img_col);
         elseif strcmp(cls_score_type, 'soft')         
           pred = pred_prob(:, :, kk+1);
           pred = sum(pred(:)) / (img_row * img_col);
         elseif strcmp(cls_score_type, 'score')
           score = raw_result(:, :, kk+1);  % 1st channle is bkg
           pred  = max(score(:));
         else
           error('Wrong cls_score_type value...\n');
         end

         cls_res_fn = sprintf('%s_cls_%s_%s.txt', cls_id, testset, VOCopts.classes{kk});

         if i == 1
           cls_fid = fopen(fullfile(save_cls_result_folder, cls_res_fn), 'w');
         else
           cls_fid = fopen(fullfile(save_cls_result_folder, cls_res_fn), 'a');
         end

         fprintf(cls_fid, '%s %f\n', img_fn, pred);

         fclose(cls_fid);
      end
    end
  end
end

% Copy the results for evaluation
CopyResultsForEvaluation(save_seg_result_folder, eval_seg_result_folder, ...
                         VOCopts, seg_id, 'seg');
CopyResultsForEvaluation(save_cls_result_folder, eval_cls_result_folder, ...
			 VOCopts, cls_id, 'cls');

% get iou score and map
if strcmp(testset, 'val')
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, seg_id);
  [recalls, precisions, aps]           = MyVOCevalcls(VOCopts, cls_id);

  fn = 'eval_seg_results.mat';
  save(fullfile(save_seg_result_folder, fn), 'accuracies', 'avacc', 'conf', 'rawcounts');

  fn = 'eval_cls_results.mat';
  save(fullfile(save_cls_result_folder, fn), 'recalls', 'precisions', 'aps');
else
  fprintf(1, 'This is test set. No evaluation. Just saved as png or txt\n');
end 
