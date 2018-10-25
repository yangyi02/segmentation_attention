anno_folder = '~/workspace/rmt/data/pascal_part/annotations/all';
anno_dir = dir(fullfile(anno_folder, '*.mat'));

save_folder = '~/workspace/rmt/data/pascal_part/annotations/person_segmentation';

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

pascal_root = '~/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages';

cmap = VOClabelcolormap();
pimap = my_part2ind();     % part index mapping

debug = 0;
missing_count = 0;

for i = 1 : numel(anno_dir)
    fprintf(1, 'processing %d (%d)\n', i, numel(anno_dir));
    
    img_fn = anno_dir(i).name(1:end-4);
      
    img = imread(fullfile(pascal_root, [img_fn, '.jpg']));
    
    % load annotation
    load(fullfile(anno_folder, anno_dir(i).name));
    
    [cls_mask, inst_mask, part_mask] = mat2map(anno, img, pimap);
    
    % save desired annotation
    labels = unique(cls_mask(:));
    
    if ismember(15, labels)
        person_seg = zeros(size(cls_mask, 1), size(cls_mask, 2), 'uint8');
        ind = cls_mask == 15;
        person_seg(ind) = 1;
        
        if debug                        
            % display annotation
            subplot(2,2,1); imshow(img); title('Image');
            subplot(2,2,4); imshow(person_seg, [0 1]); title('Part Mask');
            subplot(2,2,2); imshow(cls_mask,  cmap); title('Class Mask');
            subplot(2,2,3); imshow(inst_mask, cmap); title('Instance Mask');            
            pause;
        end
                
        part_labels = unique(part_mask(:));
        % only background exists then continue
        if length(part_labels) == 1 && ismember(0, part_labels)
            missing_count = missing_count + 1;
            continue;
        else        
            imwrite(person_seg, fullfile(save_folder, [img_fn, '.png']));
        end
    end
    
end