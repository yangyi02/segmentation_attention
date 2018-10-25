% run generate_train_val_id_list_for_person_seg.m first
%
category = 'person';

pascal_root = '~/workspace/rmt/data/pascal/VOCdevkit';
pascal_year = 'VOC2010';

list_save_folder ='~/workspace/rmt/work/deeplabel_baidu/exper/voc10_person/list';
if ~exist(list_save_folder, 'dir')
    mkdir(list_save_folder);
end

img_prefix = '/JPEGImages/';
img_postfix = '.jpg';

seg_prefix = '/SegmentationPerson/';
seg_postfix = '.png';

train_save_fn = 'train.txt';
val_save_fn   = 'val.txt';

train_id_fn = 'train_id.txt';
val_id_fn   = 'val_id.txt';

fid = fopen(fullfile(list_save_folder, train_id_fn), 'r');
train_id = textscan(fid, '%s');
train_id = train_id{1};
fclose(fid);

fid = fopen(fullfile(list_save_folder, val_id_fn), 'r');
val_id = textscan(fid, '%s');
val_id = val_id{1};
fclose(fid);

% save
fid = fopen(fullfile(list_save_folder, train_save_fn), 'w');
for i = 1 : numel(train_id)
    fprintf(fid, '%s %s\n', [img_prefix train_id{i} img_postfix], [seg_prefix train_id{i} seg_postfix]);
end
fclose(fid);

fid = fopen(fullfile(list_save_folder, val_save_fn), 'w');
for i = 1 : numel(val_id)
    fprintf(fid, '%s %s\n', [img_prefix val_id{i} img_postfix], [seg_prefix val_id{i} seg_postfix]);
end
fclose(fid);
