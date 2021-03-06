% compute the densecrf result (.bin) to png
%
WORK_ROOT_DIR = '/home/lcchen/workspace';
addpath(fullfile(WORK_ROOT_DIR, '/rmt/work/deeplabel_baidu/code/matlab/my_script'));
SetupEnv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to change values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if is_server
  if learn_crf
    post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_ModelType%d_Epoch%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, model_type, epoch);

    map_folder = fullfile(WORK_ROOT_DIR, '/rmt/work/deeplabel_baidu/exper', dataset, 'densecrf', 'res', feature_name, model_name, testset, feature_type, post_folder); 

    save_root_folder = fullfile(WORK_ROOT_DIR, '/rmt/work/deeplabel_baidu/exper', dataset, 'res', feature_name, model_name, testset, feature_type, post_folder); ;
  else
    post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std);

    map_folder = fullfile(WORK_ROOT_DIR, '/rmt/work/deeplabel_baidu/exper', dataset, 'res', feature_name, model_name, testset, feature_type, post_folder); 

    save_root_folder = map_folder;
  end
else 
  map_folder = '../result';
end

map_dir = dir(fullfile(map_folder, '*.bin'));


fprintf(1,' saving to %s\n', save_root_folder);

if strcmp(dataset, 'voc12')
  seg_res_dir = [save_root_folder '/results/VOC2012/'];
elseif strcmp(dataset, 'coco')
  seg_res_dir = [save_root_folder, '/results/COCO2014/'];
elseif strcmp(dataset, 'voc10_part')
  seg_res_dir = [save_root_folder '/results/VOC2010_part/'];
else
  error('Wrong dataset!');
end

save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

for i = 1 : numel(map_dir)
    fprintf(1, 'processing %d (%d)...\n', i, numel(map_dir));
    map = LoadBinFile(fullfile(map_folder, map_dir(i).name), 'int16');

    img_fn = map_dir(i).name(1:end-4);
    imwrite(uint8(map), colormap, fullfile(save_result_folder, [img_fn, '.png']));
end
