load('keypoints.mat');
load('im_info.mat');

root_folder = '~/workspace/rmt/data/poselets/annotations/person';

pascal_root = '~/workspace/rmt/data/pascal/VOCdevkit/VOC2012';
img_path = fullfile(pascal_root, 'JPEGImages');
config = init;
category_id = 15;  % person
K = config.K(category_id);

colors = VOClabelcolormap;

% upper
upper_points = [...
    1 2;  
    2 3;
    4 5;
    5 6;];
% lower
lower_points = [...
    7 8;  % lower
    8 9;
    9 18;
    10 11;
    11 12;
    12 19;];
% torso
torso_points = [...
    1 4;   
    1 7;
    7 10;
    4 10;];

point_link = [upper_points; lower_points; torso_points];

x_start = zeros(1, size(point_link, 1));
x_end   = zeros(1, size(point_link, 1));
y_start = zeros(1, size(point_link, 1));
y_end   = zeros(1, size(point_link, 1));

num_img = numel(annots.image_id);
num_key_points = K.NumPrimaryKeypoints;

for i = 2 : num_img
    img_fn = im.stem{annots.image_id(i)};
    img = imread(fullfile(im.image_directory, [img_fn, '.jpg']));
    
    figure(1), imshow(img), hold on;
    
    fprintf(1, 'showing image %s\n', img_fn);
    
    bbox       = annots.bounds(i, :);
    key_points = annots.coords(:, :, i);
    is_visible = annots.visible(:, i);
    
    bbox_area = bbox(3) * bbox(4);
    
    if bbox_area > 100 * 100
        p_radius = 5;
    else
        p_radius = 1;
    end
    
    for k = 1 : num_key_points
        if is_visible(k)
            color_id = 19;
        else
            color_id = 10;            
        end
    
        [r, c] = find(point_link == k);
            
        for m = 1 : numel(r)
            if c(m) == 1
                x_start(r(m)) = key_points(k, 1);
                y_start(r(m)) = key_points(k, 2);
                
            else
                x_end(r(m)) = key_points(k, 1);
                y_end(r(m)) = key_points(k, 2);
            end 
        end
           
        % mark head with smaller radius
        if (k>=13 && k<=17) || k == 20
            k_radius = 1;
        else
            k_radius = p_radius;
        end
        
        if ~isnan(key_points(k, 1)) && ~isnan(key_points(k, 2))
            viscircles([key_points(k, 1), key_points(k, 2)], ...
                    k_radius, 'EdgeColor', colors(color_id, :)); 
        end
    end
    
    for m = 1 : numel(x_start)
        if ~isnan(x_start(m)) && ~isnan(x_end(m)) && ~isnan(y_start(m)) && ~isnan(y_end(m))
            % check it is upper, lower or torso
            if m <= 4
                type = 1;
                c = colors(4, :);
            elseif m <= 10
                type = 2;
                c = colors(5, :);
            else
                type = 3;
                c = colors(3, :);
            end
            
            line([x_start(m) x_end(m)], [y_start(m) y_end(m)], 'LineWidth', 3, 'Color', c);        
        end
    end
    
    rectangle('Position', [bbox(1) bbox(2) bbox(3) bbox(4)], ...
        'EdgeColor', 'r', 'LineWidth', 3);    
    pause();    
    close(1);
end
    