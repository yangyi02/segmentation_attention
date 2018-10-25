anno_folder = '~/workspace/rmt/data/pascal_part/annotations/all';
anno_dir = dir(fullfile(anno_folder, '*.mat'));

save_folder = '~/workspace/rmt/data/pascal_part/annotations/person';
save_folder2 = '~/workspace/rmt/data/pascal_part/annotations/person_nocolor';
save_folder3 = '~/workspace/rmt/data/pascal_part/annotations/person_seg';

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end
if ~exist(save_folder2, 'dir')
    mkdir(save_folder2);
end
if ~exist(save_folder3, 'dir')
    mkdir(save_folder3);
end

pascal_root = '~/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages';

cmap = VOClabelcolormap();
pimap = my_part2ind();     % part index mapping

debug = 1;
missing_count = 0;
missing_names = {};

for i = [211 430] %1 : numel(anno_dir)
    fprintf(1, 'processing %d (%d)\n', i, numel(anno_dir));
    
    img_fn = anno_dir(i).name(1:end-4);
      
    img = imread(fullfile(pascal_root, [img_fn, '.jpg']));
    
    % load annotation
    load(fullfile(anno_folder, anno_dir(i).name));
    
    [cls_mask, inst_mask, part_mask] = mat2map(anno, img, pimap);
    
    % save desired annotation
    labels = unique(cls_mask(:));
    
    if ismember(15, labels)
        if debug
            % display annotation
            subplot(2,2,1); imshow(img); title('Image');
            subplot(2,2,2); imshow(cls_mask, cmap); title('Class Mask');
            subplot(2,2,3); imshow(inst_mask, cmap); title('Instance Mask');
            subplot(2,2,4); imshow(part_mask, cmap); title('Part Mask');
            pause;
        end
        
        part_labels = unique(part_mask(:));
        
        % only background exists then continue
        if length(part_labels) == 1 && ismember(0, part_labels)
            missing_count = missing_count + 1;
            missing_names{missing_count} = img_fn;
            continue;
        else            
            %imwrite(part_mask, cmap, fullfile(save_folder, [img_fn, '.png']));
            %imwrite(part_mask, fullfile(save_folder2, [img_fn, '.png']));
            imwrite(cls_mask, cmap, fullfile(save_folder3, [img_fn, '.png']));
        end
    end
    
end


