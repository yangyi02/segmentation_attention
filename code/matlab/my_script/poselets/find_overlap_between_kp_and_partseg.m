category = 'person';

all_kp_fn = './list/all_person_kp.txt';

part_train_fn = '../../../pascal_part/list/train_person_part_seg.txt';
part_val_fn = '../../../pascal_part/list/val_person_part_seg.txt';

save_train_overlap_fn = './list/train_person_kp_and_ps.txt';
save_val_overlap_fn   = './list/val_person_kp_and_ps.txt';
save_train_setdiff_fn = './list/person_all_kp_minus_train_ps.txt';
save_val_setdiff_fn   = './list/person_all_kp_minus_val_ps.txt';

% open files
fid = fopen(all_kp_fn, 'r');
all_kp = textscan(fid, '%s');
all_kp = all_kp{1};
fclose(fid);

fid = fopen(part_train_fn, 'r');
part_train = textscan(fid, '%s');
part_train = part_train{1};
fclose(fid);

fid = fopen(part_val_fn, 'r');
part_val = textscan(fid, '%s');
part_val = part_val{1};
fclose(fid);

all_part = [part_train; part_val];

% find overlap
train_overlap = intersect(part_train, all_kp);
val_overlap   = intersect(part_val, all_kp);

train_diff = setdiff(all_kp, part_train);
val_diff   = setdiff(all_kp, part_val);

% save lists
fid = fopen(save_train_overlap_fn, 'w');
for i = 1 : numel(train_overlap)
    fprintf(fid, '%s\n', train_overlap{i});
end
fclose(fid);

fid = fopen(save_val_overlap_fn, 'w');
for i = 1 : numel(val_overlap)
    fprintf(fid, '%s\n', val_overlap{i});
end
fclose(fid);

fid = fopen(save_train_setdiff_fn, 'w');
for i = 1 : numel(train_diff)
    fprintf(fid, '%s\n', train_diff{i});
end
fclose(fid);

fid = fopen(save_val_setdiff_fn, 'w');
for i = 1 : numel(val_diff)
    fprintf(fid, '%s\n', val_diff{i});
end
fclose(fid);

