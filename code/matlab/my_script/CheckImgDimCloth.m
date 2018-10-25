img_dir = dir('~/workspace/rmt/data/cloth_data_6/baiducloth_images/*.jpg');
min_dim = 10000 * ones(1, 2); max_dim = -1 * ones(1, 2);

for i = 1 : numel(img_dir)
    if (mod(i-1, 100) == 0)
        fprintf(1, 'processing %d (%d)\n', i, numel(img_dir));
    end
    
    img = imread(fullfile('~/workspace/rmt/data/cloth_data_6/baiducloth_images', img_dir(i).name));
    [img_row, img_col, img_chan] = size(img);
    
    if img_row < min_dim(1)
        min_dim(1) = img_row;
    end

    if img_col < min_dim(2)
        min_dim(2) = img_col;
    end

    if img_row > max_dim(1)
        max_dim(1) = img_row;
    end

    if img_col > max_dim(2)
        max_dim(2) = img_col;
    end
end

min_dim
max_dim