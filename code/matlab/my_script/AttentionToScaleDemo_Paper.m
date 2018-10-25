addpath('../export_fig/');

gpu_id = -1;  % -1 means use cpu only

%dataset = 'voc12';
%dataset = 'voc10_part';
dataset = 'coco';

WORK_ROOT_DIR = '/home/lcchen/workspace/rmt';
CAFFE_ROOT    = 'work/deeplabel_baidu/code';
WORK_DIR = fullfile('work/deeplabel_baidu/exper', dataset);
tmp=load('pascal_seg_colormap.mat');

%%% voc12
if strcmp(dataset, 'voc12')
    IMG_PATH = fullfile(WORK_ROOT_DIR, 'data/pascal/VOCdevkit/VOC2012/JPEGImages');
    
    net_name = 'vgg128_noup_pool3_20M_largewin_coco_attention10';
    attention_type = 'max';
    
    %net_name = 'vgg128_noup_pool3_20M_largewin_coco_attention13';
    %attention_type = 'cnn';

    img_names = {   
        '2007_000491'
        '2007_000783'
        '2007_001568'
        '2007_001761'
        '2007_001884'
        '2007_002046'
        '2007_006802'
        '2007_008260'
        '2010_005888'
        '2010_005344'
        '2010_005284'
        '2010_005108'
        '2010_004795'
        '2010_004789'
        '2010_004628'
        '2010_004322'
        '2010_004320'
        '2010_004120'
        '2011_000813'
        '2011_000536'
        '2011_000070'
        '2011_001232'
        '2011_001341'
        '2011_001642'
        '2011_002098'
        '2011_002121'
        '2011_002343'
        '2011_002589'
        '2011_002675'
        '2011_003256'
        '2011_003271'
        '2007_000323'
        '2011_002675'
        '2011_002322'
        '2011_001988'
        '2011_001060'
        '2011_000900'
        '2011_000248'
            '2009_001854'
          '2007_000799'
          '2007_002728'
          '2007_003188'
          '2007_003503'          
        '2007_001311'
        '2007_001630'
        '2007_005173'
        '2007_005331'
        '2007_009084'
        '2008_003461'
        '2008_007945'
        '2008_008221'
        '2009_000457'
        '2010_001024'
        };

elseif strcmp(dataset, 'voc10_part')
    IMG_PATH = fullfile(WORK_ROOT_DIR, 'data/pascal/VOCdevkit/VOC2012/JPEGImages');
%%% voc10_part
    %max: 'vgg128_noup_pool3_20M_largewin_attention30';
    net_name = 'vgg128_noup_pool3_20M_largewin_attention30'; 
    attention_type = 'max';   % 'cnn', 'max'
    
%     net_name = 'vgg128_noup_pool3_20M_largewin_attention47'; 
%     attention_type = 'cnn';   % 'cnn', 'max'

    img_names = {
        '2010_004597'
        '2010_003983'
        '2010_003632'
        '2010_003630'
        '2010_003628'
        '2010_002929'
        '2010_002927'
        '2010_002510'
        '2010_005293'
        '2010_005410'
        '2010_005626'
        '2010_005654'
        '2010_005141'
        '2010_004952'
        '2010_004786'
        '2008_000034'
        '2008_000579'
        '2008_000215'
        '2008_000492'
        '2008_000691'
        '2008_002789'
        '2008_003136'
        '2008_003228'
        '2008_003344'
        '2008_003514'
        '2008_003610'
        '2008_003825'
        '2008_005732'
        '2008_005884'
        '2008_006148'
        '2008_007585'
        '2008_000473'
        '2008_000481'
        '2008_000510'
        '2008_000522'
        '2008_000662'
        };
elseif strcmp(dataset, 'coco')
    IMG_PATH = fullfile(WORK_ROOT_DIR, 'data/coco/JPEGImages');
    
    net_name = 'vgg128_noup_pool3_20M_largewin_attention8';
    attention_type = 'max';
    
    %net_name = 'vgg128_noup_pool3_20M_largewin_attention12';
    %attention_type = 'cnn';

    img_names = {     
        'COCO_val2014_000000000042'
        'COCO_val2014_000000000294'
        'COCO_val2014_000000000675'
        'COCO_val2014_000000000923'
        'COCO_val2014_000000000985'
        'COCO_val2014_000000001089'
        'COCO_val2014_000000001146'
        'COCO_val2014_000000001532'
        'COCO_val2014_000000001592'
        'COCO_val2014_000000001626'
        'COCO_val2014_000000001799'
        'COCO_val2014_000000002235'
        'COCO_val2014_000000002255'
        'COCO_val2014_000000002562'
        'COCO_val2014_000000002972'
        'COCO_val2014_000000002985'
        'COCO_val2014_000000003109'
        'COCO_val2014_000000005107'
        'COCO_val2014_000000006847'
        'COCO_val2014_000000006896'
        'COCO_val2014_000000007088'
        'COCO_val2014_000000007115'
        'COCO_val2014_000000007304'
        'COCO_val2014_000000007682'
        'COCO_val2014_000000009007'
        'COCO_val2014_000000010056'
        'COCO_val2014_000000010123'
        'COCO_val2014_000000010205'
        'COCO_val2014_000000010693'
        'COCO_val2014_000000011202'
        'COCO_val2014_000000012192'
        'COCO_val2014_000000012748'
        'COCO_val2014_000000015301'
        'COCO_val2014_000000016285'
        'COCO_val2014_000000016977'
        'COCO_val2014_000000018444'
        'COCO_val2014_000000018462'
        'COCO_val2014_000000020254'
        'COCO_val2014_000000020774'
        'COCO_val2014_000000000136'
        'COCO_val2014_000000000192'
        'COCO_val2014_000000000241'
        'COCO_val2014_000000000328'
        'COCO_val2014_000000000395'
        'COCO_val2014_000000000428'
        'COCO_val2014_000000000459'
        'COCO_val2014_000000000488'
        'COCO_val2014_000000000536'
        'COCO_val2014_000000000599'
        'COCO_val2014_000000000623'
        'COCO_val2014_000000000632'
        'COCO_val2014_000000000641'
        'COCO_val2014_000000000692'
        'COCO_val2014_000000000757'
        'COCO_val2014_000000000761'
        'COCO_val2014_000000000923'
        'COCO_val2014_000000000974'
        'COCO_val2014_000000001000'
        'COCO_val2014_000000001153'
        'COCO_val2014_000000001164'
        'COCO_val2014_000000001228'
        'COCO_val2014_000000001268'
        'COCO_val2014_000000001532'
        'COCO_val2014_000000001584'
        'COCO_val2014_000000001626'
        'COCO_val2014_000000001840'
        'COCO_val2014_000000001869'
        'COCO_val2014_000000002014'
        'COCO_val2014_000000002235'
        'COCO_val2014_000000002239'
        };
end
%%%%%%%%%%%

if strcmp(dataset, 'voc10_part')
    num_label = 7;
    input_dim = 513;
elseif strcmp(dataset, 'voc12')
    num_label = 21;
    input_dim = 513;
elseif strcmp(dataset, 'coco')
    num_label = 91;
    input_dim = 641;
end


save_res_folder = sprintf('./paper/%s/res', dataset);
save_att_folders = cell(1, 3);
save_att_folders{1} = sprintf('./paper/%s/att1', dataset); 
save_att_folders{2} = sprintf('./paper/%s/att2', dataset); 
save_att_folders{3} = sprintf('./paper/%s/att3', dataset); 

if ~exist(save_res_folder, 'dir')
    mkdir(save_res_folder)
end
for cc = 1 : 3
    if ~exist(save_att_folders{cc}, 'dir')
        mkdir(save_att_folders{cc})
    end
end
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

for kk = 1 : numel(img_names)
    im = imread(fullfile(IMG_PATH, [img_names{kk}, '.jpg']));
    
    [img_row, img_col, img_channels] = size(im);


    % prepare oversampled input
    % input_data is Height x Width x Channel x Num
    tic;
    input_data = {PrepareImageForScaleAttentionModel(im, input_dim)};
    toc;

    % do forward pass to get scores
    % scores are now Channels x Num, where Channels == 1000
    tic;
    % The net forward function. It takes in a cell array of N-D arrays
    % (where N == 4 here) containing data of input blob(s) and outputs a cell
    % array containing data from output blob(s)
    scores = net.forward(input_data);
    toc;

    % show predicted labels
    scores = scores{1};
    scores = permute(scores, [2 1 3]);
    scores = scores(1:img_row, 1:img_col, :);
    [~, maxlabel] = max(scores, [], 3);

    % show learned attention
    if strcmp(attention_type, 'cnn')
        attention = net.blobs('attention').get_data();
        attention = permute(attention, [2 1 3]);
        attention = imresize(attention, [input_dim, input_dim], 'bilinear');
        attention = attention(1:img_row, 1:img_col, :);
        num_att = size(attention, 3);

        % figure 1 for debug
        figure(1)
        subplot(2, num_att, 1), imshow(im), title(sprintf('image %s', img_names{kk}))
        subplot(2, num_att, 2), imshow(uint8(maxlabel), tmp.colormap), title('part seg')
        freezeColors

        for c = 1 : num_att
            subplot(2, num_att, c + num_att)
            colormap(jet)
            imagesc(squeeze(attention(:,:,c)))
            colorbar, axis square
            title(sprintf('attention %d', c))
        end
        pause(1)
        
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
        
        figure(2)
        imshow(masked_img);
        save_fn = fullfile('./', save_res_folder, [img_names{kk}, '.pdf']);
        export_fig(save_fn, '-pdf', '-transparent')
        
        for cc = 1 : num_att
            figure(cc+2)
            imagesc(squeeze(attention(:,:,cc)));
            colormap(jet)
            colorbar, axis image, axis off
            save_fn = fullfile('./', save_att_folders{cc}, [img_names{kk}, '.pdf']);
            export_fig(save_fn, '-pdf', '-transparent')
        end
        
        
    elseif strcmp(attention_type, 'max')
        num_att = 3;

        score_res1 = max(net.blobs(sprintf('fc8_%s', dataset)).get_data(), [], 3);
        score_res075 = max(net.blobs(sprintf('fc8_%s_res075_interp', dataset)).get_data(), [], 3);
        score_res05 = max(net.blobs(sprintf('fc8_%s_res05_interp', dataset)).get_data(), [], 3);
        
        total_score = [score_res1(:) score_res075(:) score_res05(:)];
        [~, max_attention] = max(total_score, [], 2);
        max_attention = reshape(max_attention, size(score_res1,1), size(score_res1,2), []);
        attention = zeros(size(im, 1), size(im, 2), num_att);

        % figure 1 for debug
        figure(1)
        subplot(2, num_att, 1), imshow(im), title(sprintf('image %s', img_names{kk}))
        subplot(2, num_att, 2), imshow(uint8(maxlabel), tmp.colormap), title('part seg')
        freezeColors
        for c = 1 : num_att
            tmp2 = zeros(size(max_attention, 1), size(max_attention, 2));
            tmp2(max_attention == c) = 1;
            tmp2 = permute(tmp2, [2 1 3]);
            tmp2 = imresize(tmp2, [input_dim, input_dim], 'nearest');
            
            attention(:, :, c) = tmp2(1:img_row, 1:img_col);

            subplot(2, num_att, c + num_att)
            colormap(jet)
            imagesc(squeeze(attention(:,:,c)))
            colorbar, axis square
            title(sprintf('attention %d', c))
        end        
        %pause()
        
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
        
        figure(2)
        imshow(masked_img);
        save_fn = fullfile('./', save_res_folder, [img_names{kk}, '_max.pdf']);
        export_fig(save_fn, '-pdf', '-transparent')
        
        for cc = 1 : num_att
            figure(cc+2)
            imagesc(squeeze(attention(:,:,cc)));
            colormap(jet)
            colorbar, axis image, axis off
            save_fn = fullfile('./', save_att_folders{cc}, [img_names{kk}, '_max.pdf']);
            export_fig(save_fn, '-pdf', '-transparent')
        end
    else
        error('not supported attention type\n');
    end
end
% call caffe.reset_all() to reset caffe
caffe.reset_all();
