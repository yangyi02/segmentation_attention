gpu_id = -1;  % -1 means use cpu only
dataset = 'voc10_part';
num_label = 7;

file_folder = './jiang/orig';
frames_folder = './jiang/frames';
res_folder = './jiang/res';
res_video_folder = './jiang/res/video';

do_extract_frames = 0;
do_part_parsing   = 1;
do_save_res_as_video = 1;

if do_extract_frames
    % extract frames from video
    movie = VideoReader(fullfile(file_folder, 's14-d50_1059_1136.avi'));
    num_frames = movie.NumberOfFrames;

    for ii = 1 : num_frames
        frame = read(movie, ii);
        imwrite(frame, fullfile(frames_folder, sprintf('%03d.jpg', ii)));
    end
end

if do_part_parsing
    % set up network
    WORK_ROOT_DIR = '/home/lcchen/workspace/rmt';
    CAFFE_ROOT    = 'work/deeplabel_baidu/code';

    WORK_DIR = fullfile('work/deeplabel_baidu/exper', dataset);
    tmp=load('pascal_seg_colormap.mat');

    net_name = 'vgg128_noup_pool3_20M_largewin_attention47'; 

    input_dim = 513;    %481;   %513;    

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
    model  = fullfile(WORK_ROOT_DIR, WORK_DIR, 'model', net_name, 'train_iter_6000.caffemodel');
    phase = 'test';
    net = caffe.Net(config, model, phase);

    frame_dir = dir(fullfile(frames_folder, '*.jpg'));
   
    for ii = 1 : numel(frame_dir)
        im = imread(fullfile(frames_folder, frame_dir(ii).name));
        im = my_imresize(im, input_dim);
        [img_row, img_col, img_channels] = size(im);

        input_data = {PrepareImageForScaleAttentionModel(im, input_dim)};
        scores = net.forward(input_data);

        % show predicted labels
        scores = scores{1};
        scores = permute(scores, [2 1 3]);
        scores = scores(1:img_row, 1:img_col, :);
        [~, maxlabel] = max(scores, [], 3);
        
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
        
        h = figure(2);
        subplottight(1, 3, 1); imshow(im), title('image')
        subplottight(1, 3, 2); imshow(maxlabel, tmp.colormap), title('part seg')            
        freezeColors
        subplottight(1, 3, 3); imshow(masked_img), title('result')
        
        save_fn = fullfile(res_folder, sprintf('%03d.jpg', ii));
        saveas(h, save_fn, 'jpg');
    end    
end

if do_save_res_as_video
    fn = fullfile(res_video_folder, 's14-d50_1059_1136.avi');
    writerObj = VideoWriter(fn);
    writerObj.FrameRate = 30;
    open(writerObj);
    
    res_dir = dir(fullfile(res_folder, '*.jpg'));
    
    for jj = 1 : numel(res_dir)
        img = imread(fullfile(res_folder, res_dir(jj).name));
        writeVideo(writerObj, img);
    end
    close(writerObj);    
end

% call caffe.reset_all() to reset caffe
caffe.reset_all();
