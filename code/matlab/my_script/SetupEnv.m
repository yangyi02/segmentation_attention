% set up the environment variables
%
clear all; close all;
load('./pascal_seg_colormap.mat');
WORK_ROOT_DIR = '/home/lcchen/workspace'; 

%
is_server       = 1;

crf_load_mat    = 1;   % the densecrf code load MAT files directly (no call SaveMatAsBin.m)
                       % used ONLY by DownSampleFeature.m
learn_crf       = 0;   % is the crf parameters learned or cross-validated

is_mat          = 0;   % the results to be evaluated are saved as mat or png
has_postprocess = 0;   % has done densecrf post processing or not
is_argmax       = 1;   % the output has been taken argmax already (e.g., coco dataset). 
                       % assume the argmax takes C-convention (i.e., start from 0)

debug           = 0;   % if debug, show some results

% crf parameters
%%%%%%%%%%%%%%%%%%%%%% voc10_part %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% vgg128_noup_pool3_20M_largewin2
% bi_w = 5; bi_x_std = 13; bi_r_std = 3; pos_w = 3; pos_x_std = 3;

% vgg128_ms_pool3_20M_largewin2
% bi_w = 5; bi_x_std = 9; bi_r_std = 5; pos_w = 3; pos_x_std = 3;

% deconv_exp17
% bi_w = 4; bi_x_std = 10; bi_r_std = 4; pos_w = 3; pos_x_std = 3;

% deconv_ms12
% bi_w = 3; bi_x_std = 19; bi_r_std = 3; pos_w = 3; pos_x_std = 3;

% vgg128_noup_pool3_20M_largewin_attention30
% bi_w = 4; bi_x_std = 15; bi_r_std = 3; pos_w = 3; pos_x_std = 3;

% vgg128_noup_pool3_20M_largewin_attention28_2
% bi_w = 5; bi_x_std = 11; bi_r_std = 5; pos_w = 3; pos_x_std = 3;

% vgg128_noup_pool3_20M_largewin_attention46
% bi_w = 4; bi_x_std = 20; bi_r_std = 5; pos_w = 3; pos_x_std = 3;

% vgg128_noup_pool3_20M_largewin_attention47
% bi_w = 4; bi_x_std = 19; bi_r_std = 3; pos_w = 3; pos_x_std = 3;

% vgg128_noup_pool3_20M_largewin4
% bi_w = 4; bi_x_std = 21; bi_r_std = 3; pos_w = 3; pos_x_std = 3;

% vgg128_ms_pool3_20M_largewin4
% bi_w = 4; bi_x_std = 21; bi_r_std = 3; pos_w = 3; pos_x_std = 3;

% vgg128_ms_pool3_20M_largewin_attention9_2
% bi_w = 4; bi_x_std = 9; bi_r_std = 5; pos_w = 3; pos_x_std = 3;

% vgg128_ms_pool3_20M_largewin_attention13
% bi_w = 4; bi_x_std = 19; bi_r_std = 3; pos_w = 3; pos_x_std = 3;

% vgg128_ms_pool3_20M_largewin_attention30
% bi_w = 4; bi_x_std = 13; bi_r_std = 3; pos_w = 3; pos_x_std = 3;


%%%%%%%%%%%%%%%%%%%%%%%%%% voc12 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% vgg128_noup_pool3_20M_largewin_coco_attention13
% bi_w = 4; bi_x_std = 49; bi_r_std = 5; pos_w = 3; pos_x_std = 3;

% vgg128_ms_pool3_20M_largewin_coco_attention9
% bi_w = 5; bi_x_std = 51; bi_r_std = 5; pos_w = 3; pos_x_std = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% initial or default values for crf
bi_w           = 4; 
bi_x_std       = 49;
bi_r_std       = 5;

pos_w          = 3;
pos_x_std      = 3;

% used for learn_crf
model_type     = 1;  % 0: Potts, 1: Diagonal, 2: Matrix label compacitability
epoch          = 1;  % used for learned crf parameters
%

%%%%%%%%%%%%%%%%%%%%%%%%%%
%dataset    = 'voc10_part';  
%model_name = 'vgg128_noup_pool3_20M_largewin_attention28_3';

%dataset    = 'voc12';  
%model_name = 'vgg128_noup_pool3_20M_largewin_coco_attention11_2';
%model_name = 'vgg128_noup_pool3_20M_largewin_attention12';
%model_name = 'vgg128_ms_pool3_20M_largewin_attention4';

%dataset = 'cloth_data_6';
%model_name = 'vgg128_noup_pool3_20M_largewin_attention1';

dataset = 'coco';
model_name = 'vgg128_ms_pool3_20M_largewin';
%model_name = 'vgg128_noup_pool3_20M_largewin_attention12';
%model_name = 'vgg128_ms_pool3_20M_largewin_attention12';

%%%%%%%%%%%%%%%%%%%%%%%%%%
trainset   = 'train_aug';       % not used
testset    = 'val';            %'val', 'test'

feature_name = 'features';
%feature_name = 'features2';
feature_type = 'fc8'; % fc8 / crf / fc8_spm / fc7_spm

multi_feature_type = {'seg_score', 'seg_score_res2', 'seg_score_res4'}; %, 'seg_score_res8'};

% used for multiscale inputs
gpu_id = 1;
input_max_size = 617;   % 721, 617
input_scales   = [0.8, 1, 1.2];
deploy_file_name = sprintf('deploy_inputsize_%d.prototxt', input_max_size);

if strcmp(feature_name, 'features')
  weight_file_name = 'train_iter_6000.caffemodel';    % used for deploy.txt
elseif strcmp(feature_name, 'features2')
  weight_file_name = 'train2_iter_8000.caffemodel';
end

% method to get "scores" for image classification task
cls_score_type = 'score';    % 'hard', 'soft', 'score'

% feature_name = 'erode_gt';     % 'erode_gt', 'features', 'features4', 'features2', ''
% feature_type = 'bboxErode20_OccluBias';        %'bboxErode20', 'fc8', 'crf', 'fc8_crf'


id                 = 'comp6';
seg_id             = id;
seg_task_folder    = 'Segmentation';
seg_gt_task_folder = 'SegmentationClass';

cls_id             = 'comp2';
cls_task_folder    = 'Main';
cls_gt_task_folder = 'Main';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% used for cross-validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(10)

% downsampling files for cross-validation
down_sample_method = 2;      % 1: equally sample, 2: randomly pick num_sample
down_sample_rate   = 8;
num_sample         = 100;     % used for erode_gt 

% ranges for cross-validation
range_bi_w = [4 6];
range_bi_x_std = [51];
range_bi_r_std = [5];

range_pos_w = [3];
range_pos_x_std = [3];

