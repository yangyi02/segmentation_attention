category = 'person';

pascal_root = '~/workspace/rmt/data/pascal/VOCdevkit';
pascal_year = 'VOC2010';

annot_root  = '~/workspace/rmt/data/pascal_part/annotations';

list_save_folder ='~/workspace/rmt/work/deeplab_baidu/exper/voc10_part/list';
if ~exist(list_save_folder, 'dir')
    mkdir(list_save_folder);
end

train_save_fn = 'train_person_part_seg_id.txt';
val_save_fn   = 'val_person_part_seg_id.txt';

% open pascal original list
pascal_train_fn = fullfile(pascal_root, pascal_year, 'ImageSets', 'Main', ...
    'train.txt');
pascal_val_fn = fullfile(pascal_root, pascal_year, 'ImageSets', 'Main', ...
    'val.txt');
%pascal_train_fn = fullfile(pascal_root, pascal_year, 'ImageSets', 'Main', ...
%    [category '_train.txt']);
%pascal_val_fn = fullfile(pascal_root, pascal_year, 'ImageSets', 'Main', ...
%    [category '_val.txt']);

fid = fopen(pascal_train_fn, 'r');
%pascal_train_list = textscan(fid, '%s %d');
pascal_train_list = textscan(fid, '%s');
%has_object = cell2mat(pascal_train_list(2));
pascal_train_list = pascal_train_list{1};
%pascal_train_list = pascal_train_list(has_object == 1);
fclose(fid);

fid = fopen(pascal_val_fn, 'r');
%pascal_val_list = textscan(fid, '%s %d');
pascal_val_list = textscan(fid, '%s');
%has_object = cell2mat(pascal_val_list(2));
pascal_val_list = pascal_val_list{1};
%pascal_val_list = pascal_val_list(has_object == 1);
fclose(fid);

% find all person part segmentation list
part_dir = dir(fullfile(annot_root, category, '*.png'));

% create desired train/val list
train_list = cell(1, numel(part_dir));
num_train = 0;
val_list = cell(1, numel(part_dir));
num_val = 0;

for i = 1 : numel(part_dir)
    fn = part_dir(i).name(1:end-4);
    
    if ismember(fn, pascal_train_list)
        num_train = num_train + 1;
        train_list{num_train} = fn;
    elseif ismember(fn, pascal_val_list)
        num_val = num_val + 1;
        val_list{num_val} = fn;
    end
end

% save lists
fid = fopen(fullfile(list_save_folder, train_save_fn), 'w');
for i = 1 : num_train
    fprintf(fid, '%s\n', train_list{i});
end
fclose(fid);

fid = fopen(fullfile(list_save_folder, val_save_fn), 'w');
for i = 1 : num_val
    fprintf(fid, '%s\n', val_list{i});
end
fclose(fid);



