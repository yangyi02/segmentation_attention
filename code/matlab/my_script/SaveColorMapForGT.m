gt_folder = '~/workspace/rmt/work/deeplabel_baidu/paper/cvpr2016/fig/coco/gt';
tmp = load('pascal_seg_colormap.mat');

gt_dir = dir(fullfile(gt_folder, '*.png'));

save_folder = '~/workspace/rmt/work/deeplabel_baidu/paper/cvpr2016/fig/coco/gt_color';

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

for ii = 1 : numel(gt_dir)
    gt = imread(fullfile(gt_folder, gt_dir(ii).name));
    imwrite(gt, tmp.colormap, fullfile(save_folder, gt_dir(ii).name));
end