clear all; close all; clc;

gpu_id = 1;  % -1 means use cpu only
dataset = 'voc10_part';
%dataset = 'cloth_data_6';

WORK_ROOT_DIR = '/home/lcchen/workspace/rmt';
CAFFE_ROOT    = 'work/deeplabel_baidu/code';
WORK_DIR = fullfile('work/deeplabel_baidu/exper', dataset);

do_extract_frames = 0;

if strcmp(dataset, 'voc10_part')
    num_label = 7;
    frame_folder = './frames_part';
    net_name = 'vgg128_noup_pool3_20M_largewin_attention47'; 
    
    IMG_PATH = fullfile(WORK_ROOT_DIR, 'data/mpii/video');
    img_folders = {
        '1/000506158'
        '1/003118314'
        '1/007513778'            
        '2/000575866'
        '2/001787393'
        '2/003934354'        
        '2/003942681'        
        '3/000156511'
        '3/000381992'
%         
%     '1/008023375'    
%     '2/000638587'
%     '2/001212077'
%     '2/002440078'
%     '2/002619671'
%     '2/003934354'    
    };
    res_folder = 'res_part';
    tmp=load('pascal_seg_colormap.mat');
    
    input_dim = 513;    %481;   %513;
    model  = fullfile(WORK_ROOT_DIR, WORK_DIR, 'model', net_name, 'train_iter_6000.caffemodel');
    
elseif strcmp(dataset, 'cloth_data_6')
    num_label = 6;
    frame_folder = './frames_cloth';
    net_name = 'vgg128_noup_pool3_20M_largewin_attention1';
    
    %IMG_PATH = fullfile(WORK_ROOT_DIR, 'data/cloth_data_6/video');
    IMG_PATH = './frames_cloth';
    
    img_folders = {'1'}; {'3', '4', '5', '6', '7', '8'};
    res_folder = 'res_cloth';
    
    tmp.colormap = zeros(6, 3);
    tmp.colormap(2, :) = [255 255 63] / 255;
    tmp.colormap(3, :) = [63 63 255] / 255;
    tmp.colormap(4, :) = [255 63 63] / 255;
    tmp.colormap(5, :) = [255 63 255] / 255;
    tmp.colormap(6, :) = [63 255 255] / 255;
    
    input_dim = 481;    %481;   %513;    
    model  = fullfile(WORK_ROOT_DIR, WORK_DIR, 'model', net_name, 'train_iter_12000.caffemodel');
end

debug = 1;

if ~exist(frame_folder, 'dir')
    mkdir(frame_folder);
end


if do_extract_frames
    
    for kk = 1 : numel(img_folders)
        video_folder = fullfile('./videos_cloth', img_folders{kk});
        
        if ~exist(fullfile(frame_folder, img_folders{kk}), 'dir')
            mkdir(fullfile(frame_folder, img_folders{kk}));
        end
        
        % extract frames from video
        %movie = VideoReader(fullfile(video_folder, 's14-d50_1059_1136.avi'));
        movie = VideoReader(fullfile(video_folder, '001.avi'));

        num_frames = movie.NumberOfFrames;
        for ii = 1 : num_frames
            frame = read(movie, ii);
            imwrite(frame, fullfile(frame_folder, img_folders{kk}, sprintf('%04d.jpg', ii)));
        end
    end
end    
    


attention_type = 'cnn';   % 'cnn', 'max'

if exist(fullfile(WORK_ROOT_DIR, CAFFE_ROOT, 'matlab/+caffe'), 'dir')
  addpath(fullfile(WORK_ROOT_DIR, CAFFE_ROOT, 'matlab'));
else
  error('Cannot find CAFFE')
end

% Set caffe mode
if gpu_id >= 0
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

config = fullfile(WORK_ROOT_DIR, WORK_DIR, 'config', net_name, 'deploy.prototxt');


phase = 'test';
net = caffe.Net(config, model, phase);

for kk = 1 : numel(img_folders)
    img_folder = img_folders{kk};
    
    % set up video
    video_name = strrep(img_folder, '/', '_');
    save_folder = fullfile(res_folder, video_name);
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end

    % get images
    img_dir =dir(fullfile(IMG_PATH, [img_folder, '/*.jpg']));
    images  = cell(1, numel(img_dir));
    masked_images = cell(1, numel(img_dir));
    results = cell(1, numel(img_dir));
    scales  = cell(3, numel(img_dir));

    for ii = 1 : numel(img_dir)
        im = imread(fullfile(IMG_PATH, img_folder, img_dir(ii).name));
        im = my_imresize(im, input_dim);

        [img_row, img_col, img_channels] = size(im);

        input_data = {PrepareImageForScaleAttentionModel(im, input_dim)};
        scores = net.forward(input_data);

        % show predicted labels
        scores = scores{1};
        scores = permute(scores, [2 1 3]);
        scores = scores(1:img_row, 1:img_col, :);
        [~, maxlabel] = max(scores, [], 3);

        images{ii} = input_data{1};
        results{ii} = maxlabel;

        masked_img = im;
        for cc = 1 : num_label
            ind = maxlabel == cc;

            for c = 1 : 3
                gt_color = tmp.colormap(cc, c) * 255;
                masked_img_channel = masked_img(:, :, c);
                masked_img_channel(ind) = 0.5 * masked_img_channel(ind) + 0.5 * gt_color;
                masked_img(:,:,c) = masked_img_channel;
            end
        end
        masked_images{ii} = masked_img;

        % show learned attention
        if strcmp(attention_type, 'cnn')
            attention = net.blobs('attention').get_data();
            attention = permute(attention, [2 1 3]);
            attention = imresize(attention, 8, 'bilinear');
            attention = attention(1:img_row, 1:img_col, :);
            num_att = size(attention, 3);

            assert(num_att == 3);

            if debug
                h = figure(2);
                subplottight(1, num_att, 1); imshow(im), title('image')
                subplottight(1, num_att, 2); imshow(maxlabel, tmp.colormap), title('part seg')            
                freezeColors
                subplottight(1, num_att, 3); imshow(masked_img), title('result')
                
%                 subplottight(2, num_att, 1); imshow(im), title('image')
%                 subplottight(2, num_att, 2); imshow(maxlabel, tmp.colormap), title('part seg')            
%                 freezeColors
%                 subplottight(2, num_att, 3); imshow(masked_img), title('result')
%                 for c = 1 : num_att                
%                     subplottight(2, num_att, c + num_att);
%                     cjet = colormap(jet(256));
%                     att = squeeze(attention(:,:,c));
%                     att_min = min(att(:));
%                     att_max = max(att(:));
%                     att = (att - att_min) / (att_max - att_min);
%                     %imagesc(squeeze(attention(:,:,c)))
%                     imshow(uint8(att*255), cjet);
%                     %colorbar, axis image
% 
%                     if c == 1
%                         title('scale 1');
%                     elseif c == 2
%                         title('scale 0.75');
%                     elseif c == 3
%                         title('scale 0.5');
%                     else
%                         error('wrong c')
%                     end                    
%                 end
            end
            for c = 1 : num_att
                scales{c, ii} = squeeze(attention(:,:,c));            
            end
        elseif strcmp(attention_type, 'max')
            %%%%%%%%%%%%% not done yet
            num_att = 4;

            score_res1 = max(net.blobs('fc8_voc10_part').get_data(), [], 3);
            score_res075 = max(net.blobs('fc8_voc10_part_res075_interp').get_data(), [], 3);
            score_res05 = max(net.blobs('fc8_voc10_part_res05_interp').get_data(), [], 3);
            score_res025 = max(net.blobs('fc8_voc10_part_res025_interp').get_data(), [], 3);
            total_score = [score_res1(:) score_res075(:) score_res05(:) score_res025(:)];
            [~, max_attention] = max(total_score, [], 2);
            max_attention = reshape(max_attention, size(score_res1,1), size(score_res1,2), []);
            attention = zeros(size(max_attention, 1), size(max_attention, 2), 4);

            figure(1)
            subplot(2, num_att, 1), imshow(im), title(sprintf('image %s', img_name))
            subplot(2, num_att, 2), imshow(uint8(maxlabel), tmp.colormap), title('part seg')
            freezeColors
            for c = 1 : num_att
                tmp = zeros(size(max_attention, 1), size(max_attention, 2));
                tmp(max_attention == c) = 1;
                attention(:, :, c) = tmp;

                subplot(2, num_att, c + num_att)
                colormap(jet)
                imagesc(squeeze(attention(:,:,c)))
                colorbar, axis square
                title(sprintf('attention %d', c))
            end
        else
            error('not supported attention type\n');
        end

        save_fn = fullfile(save_folder, sprintf('%03d.jpg', ii));
        saveas(h, save_fn, 'jpg');
    end
end   


    

% call caffe.reset_all() to reset caffe
caffe.reset_all();
