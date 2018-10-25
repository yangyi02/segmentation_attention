pascal_root= '~/workspace/rmt/data/pascal/VOCdevkit/VOC2012/';

%%% part and person seg
%part
dataset1 = 'voc10_part';
model1 = 'deconv_exp17';

%person
dataset2 = 'voc10_person';
model2 = 'deconv_exp1';

res_root1 = fullfile('/home/lcchen/workspace/rmt/work/deeplabel_baidu/exper', dataset1,...
    'res/features');
res_root2 = fullfile('/home/lcchen/workspace/rmt/work/deeplabel_baidu/exper', dataset2,...
    'res/features');

seg1 = 'seg_score';
seg2 = 'seg_score';

seg_folders1 = 'post_none/results/VOC2010_part/Segmentation/comp6_val_cls';
seg_folders2 = 'post_none/results/VOC2010_person/Segmentation/comp6_val_cls';

val_res_dir = dir(fullfile(res_root1, model1, 'val', seg1, seg_folders1, '*.png'));

for i = 1 : numel(val_res_dir)
    img_fn = val_res_dir(i).name(1:end-4);
    
    img = imread(fullfile(pascal_root, 'JPEGImages', [img_fn, '.jpg']));
    [gt, colormap] = imread(fullfile(pascal_root, 'SegmentationPart_Visualization', [img_fn, '.png']));
    res1 = imread(fullfile(res_root1, model1, 'val', seg1, seg_folders1, [img_fn, '.png']));
    res2 = imread(fullfile(res_root2, model2, 'val', seg2, seg_folders2, [img_fn, '.png']));
    
    figure(1),
    subplot(2,2,1), imshow(img)
    subplot(2,2,2), imshow(gt, colormap)
    subplot(2,2,3), imshow(res1, colormap), title('part')
    subplot(2,2,4), imshow(res2, colormap), title('frg/bkg')
    pause()
end



%%% deconv-bn with mean and wo mean, deeplab
% model1 = 'deconv_exp1';
% model2 = 'vgg128_noup_pool3_20M_largewin2';
% 
% res_root = '/home/lcchen/workspace/rmt/work/deeplabel_baidu/exper/voc10_part/res/features';
% 
% seg1 = 'seg_score_centerMean_noMovingAvg';
% seg2 = 'seg_score_centerMean_hasMovingAvg';
% seg3 = 'fc8';
% 
% seg_folders = 'post_none/results/VOC2010_part/Segmentation/comp6_val_cls';
% 
% res_dir = dir(fullfile(res_root, model1, 'val', seg1, seg_folders, '*.png'));
% 
% for i = 1 : numel(res_dir)
%     img_fn = res_dir(i).name(1:end-4);
%     
%     img = imread(fullfile(pascal_root, 'JPEGImages', [img_fn, '.jpg']));
%     [gt, colormap] = imread(fullfile(pascal_root, 'SegmentationPart_Visualization', [img_fn, '.png']));
%     res1 = imread(fullfile(res_root, model1, 'val', seg1, seg_folders, [img_fn, '.png']));
%     res2 = imread(fullfile(res_root, model1, 'val', seg2, seg_folders, [img_fn, '.png']));
%     res3 = imread(fullfile(res_root, model2, 'val', seg3, seg_folders, [img_fn, '.png']));
%     
%     figure(1),
%     subplot(2,3,1), imshow(img)
%     subplot(2,3,2), imshow(gt, colormap)
%     subplot(2,3,4), imshow(res1, colormap), title('deconv-noAvg')
%     subplot(2,3,5), imshow(res2, colormap), title('deconv-hasAvg')
%     subplot(2,3,6), imshow(res3, colormap), title('deeplab')
%     pause()
% end