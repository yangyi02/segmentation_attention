% evaluate the classification performance with svm
% need to call ConcatenateFeatures.m to get the feature matrix
%

SetupEnv;

addpath('/rmt/work/deeplabel/code/ext/libsvm-3.20/matlab');
addpath('/rmt/work/deeplabel/code/ext/liblinear-1.96/matlab');

%% prepare to train SVM

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

train_mat_folder = fullfile('/rmt/work/deeplabel/exper', dataset, feature_name, model_name, trainset, feature_type);

test_mat_folder = fullfile('/rmt/work/deeplabel/exper', dataset, feature_name, model_name, testset, feature_type);


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

% load feature and lists
tmp = load(fullfile(train_mat_folder, 'concat', 'feat_matrix_and_label.mat'));
features = tmp.feat;
labels   = tmp.labels;

num_class = numel(VOCopts.classes);

% start to train svm
use_libsvm = 1;   % 0: use liblinear, 1: use libsvm
use_L2 = 1;
svm_c = 1;
svm_e = 0.001;
svm_b = -1;         %bias: -1: no bias

if use_L2
  kermap = @(x) bsxfun(@rdivide, x, sqrt(sum(x.^2)));  % L2-norm
else
  kermap = @(x) x;
end


model_fn = sprintf('models_libsvm%d_L2%d_C%.1f_bias%d.mat', use_libsvm, use_L2, svm_c, svm_b);

if ~exist(fullfile(train_mat_folder, 'models'), 'dir')
  mkdir(fullfile(train_mat_folder, 'models'));
end

if exist(model_fn, 'file')
  fprintf(1, 'models have been trained...loading...\n');
  tmp = load(fullfile(train_mat_folder, 'models', model_fn));
  models = tmp.models;
  clear tmp;
else
  models = cell(1, num_class);

  % train one-vs-all SVM
  train_time = zeros(1, num_class);
  for k = 1 : num_class
    fprintf(1, 'training svm for class %s...', VOCopts.classes{k});
 
    tic;
    if use_libsvm
      models{k} = svmtrain( labels(k,:)', kermap(features)', sprintf('-s 0 -t 0 -e %f -c %f -q', svm_e, svm_c));
    else
      models{k} = train(labels(k,:)', sparse(kermap(features)'), sprintf('-s 3 -e %f -c %f -B %f -q', svm_e, svm_c, svm_b));
    end

    train_time(k) = toc;
    fprintf(1, 'takes %f sec...', train_time(k));

    % get training accuracy for debug
    if use_libsvm
      [pred_label, accuracy, decision_values] = ...
         svmpredict( labels(k,:)', kermap(features)', models{k});    
    else
      [pred_label, accuracy, decision_values] = ...
  	 predict( labels(k,:)', sparse(kermap(features)'), models{k});    
    end      
  end
  fprintf(1, 'totoal training time %f sec...\n', sum(train_time));
  fprintf(1, 'saving trained models at %s...\n', train_mat_folder);
  save(fullfile(train_mat_folder, 'models', model_fn), 'models');
end

% evaluate the models
tmp = load(fullfile(test_mat_folder, 'concat', 'feat_matrix_and_label.mat'));
features = tmp.feat;

[feat_dim, num_img] = size(features);

if strcmp(testset, 'val')
  labels   = tmp.labels;
elseif strcmp(testset, 'test')
  labels   = rand(num_class, num_img);
end
clear tmp;

scores = zeros(size(labels));
test_time = zeros(1, num_class);

for k = 1 : num_class
  fprintf(1, 'evaluating class %s\n', VOCopts.classes{k});

  tic;  
  if use_libsvm
    [pred_label, accuracy, decision_values] = ...
         svmpredict( labels(k,:)', kermap(features)', models{k});
  else
    [pred_label, accuracy, decision_values] = ...
  	 predict( labels(k,:)', sparse(kermap(features)'), models{k});    
   end
  test_time(k) = toc;
 
  scores(k, :) = decision_values;
end
fprintf(1, 'total test time %f sec...\n', sum(test_time));


%% use PASCAL evaluation scripts
% iterate through all images (note cls and seg have non-overlapped sets)
test_dir = dir(fullfile(test_mat_folder, '*.mat'));
assert(numel(test_dir) == num_img);

for k = 1 : num_class
  cls_res_fn = sprintf('%s_cls_%s_%s.txt', cls_id, testset, VOCopts.classes{k});

  cls_fid = fopen(fullfile(save_cls_result_folder, cls_res_fn), 'w');
  
  for m = 1 : num_img
      img_fn = test_dir(m).name(1:end-4);
      img_fn = strrep(img_fn, '_blob_0', '');

      fprintf(cls_fid, '%s %f\n', img_fn, scores(k, m));
  end
  
  fclose(cls_fid);
end    

% Copy the results for evaluation
CopyResultsForEvaluation(save_cls_result_folder, eval_cls_result_folder, ...
			 VOCopts, cls_id, 'cls');

% get iou score and map
if strcmp(testset, 'val')
  [recalls, precisions, aps] = MyVOCevalcls(VOCopts, cls_id);

  fn = 'eval_cls_results.mat';
  save(fullfile(save_cls_result_folder, fn), 'recalls', 'precisions', 'aps');
else
  fprintf(1, 'This is test set. No evaluation. Just saved as png or txt\n');
end 

