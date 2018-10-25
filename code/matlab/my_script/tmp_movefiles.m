src_folder = '/rmt/work/deeplabel/exper/voc12/features2/vgg128_noup_pool3_20M_largewin3_coco/val/fc8';
dest_folder = '/rmt/work/deeplabel/exper/voc12/features2/vgg128_noup_pool3_20M_largewin3_coco/val/crf';

crf_folder = '/rmt/work/deeplabel/exper/voc12/features2/vgg128_noup_pool3_20M_largewin3_coco_cls_baseline/val/crf'; 

src_dir = dir(fullfile(src_folder, '*.mat'));

for i = 1 : numel(src_dir)
  fprintf(1, 'processing %d (%d) ...\n', i, numel(src_dir));
  src_fn = fullfile(crf_folder, src_dir(i).name);
  dest_fn = fullfile(dest_folder, src_dir(i).name);
  copyfile(src_fn, dest_fn);
end
