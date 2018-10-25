category = 'person';

pascal_root = '~/workspace/rmt/data/pascal/VOCdevkit';
pascal_year = 'VOC2011';

load('keypoints.mat');
load('im_info.mat');

annot_root = '../../annotations';

list_save_folder ='./kp_annotations';
if ~exist(list_save_folder, 'dir')
    mkdir(list_save_folder);
end

train_save_fn = 'train_person_kp.txt';
val_save_fn   = 'val_person_kp.txt';

% open pascal original list
pascal_train_fn = fullfile(pascal_root, pascal_year, 'ImageSets', 'Main', ...
    'train.txt');
pascal_val_fn = fullfile(pascal_root, pascal_year, 'ImageSets', 'Main', ...
    'val.txt');

fid = fopen(pascal_train_fn, 'r');
pascal_train_list = textscan(fid, '%s');
pascal_train_list = pascal_train_list{1};
fclose(fid);

fid = fopen(pascal_val_fn, 'r');
pascal_val_list = textscan(fid, '%s');
pascal_val_list = pascal_val_list{1};
fclose(fid);

% find all person part segmentation list
% it is saved as im.stem
kp_list = im.stem;

% create desired train/val list
train_list = cell(1, numel(kp_list));
num_train = 0;
val_list = cell(1, numel(kp_list));
num_val = 0;
miss_list = cell(1, numel(kp_list));
num_miss = 0;

for i = 1 : numel(kp_list)
    fn = kp_list{i};
    
    if ismember(fn, pascal_train_list)
        num_train = num_train + 1;
        train_list{num_train} = fn;
    elseif ismember(fn, pascal_val_list)
        num_val = num_val + 1;
        val_list{num_val} = fn;
    else
        num_miss = num_miss + 1;
        miss_list{num_miss} = fn;
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



