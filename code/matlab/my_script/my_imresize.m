function out = my_imresize(img, max_img_dim)
% resize input img so that that max dim = max_img_dim
%
[img_row, img_col, ~] = size(img);

% find max dim
max_dim = img_row;
if max_dim < img_col;
    max_dim = img_col;
end
    
if max_dim > max_img_dim
    resize_scale = max_img_dim / max_dim;
else
    resize_scale = 1;
end
    
if resize_scale ~= 1
    out = imresize(img, resize_scale, 'bilinear');
else
    out = img;    
end

[img_row, img_col, ~] = size(out);

if img_row > max_img_dim
    new_img_row = max_img_dim;
else
    new_img_row = img_row;
end

if img_col > max_img_dim
    new_img_col = max_img_dim;
else
    new_img_col = img_col;
end

out = out(1:new_img_row, 1:new_img_col, :);