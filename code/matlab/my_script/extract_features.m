%% INFO
% George Papandreou, TTI-C, February 2014
% Extract Caltech 101/256 features using a deep neural net similar to
% Krizhevsky's, trained using Caffe on the Imagenet LSVRC-2010 or LSVRC-2012 image
% classification dataset.
% Test the performance of these features on the Caltech task.

%% Set up network
disp('Deep Net Setup');

addpath('/home/gpapan/code/caffe/matlab/caffe');

net_dir = '/rmt/work/caffe/imagenet';

model_instance = 8;
switch model_instance
  case 1
    net_id = 'alexnet'; layer = 'fc7';
    model_file = fullfile(net_dir,'2010','model',net_id,'alexnet_imagenet_train_iter_1000000');
    config_file = fullfile(net_dir,'2010','config',net_id,sprintf('deploy_%s.prototxt',layer));
    mean_file = fullfile(net_dir,'2010','protobuf/train_mean_pca.bproto');
    im_size = [226 226 3]; feat_dim = 4096;
  case 2
    net_id = 'epit2'; layer = 'fc7';
    model_file = fullfile(net_dir,'2010','model',net_id,'train_iter_550000');
    config_file = fullfile(net_dir,'2010','config',net_id,sprintf('deploy_%s.prototxt',layer));
    mean_file = fullfile(net_dir,'2010','protobuf/train_mean_pca.bproto');
    im_size = [221 221 3]; feat_dim = 4096;
  case 3
    net_id = 'alexnet2'; layer = 'fc7';
    model_file = fullfile(net_dir,'2012','model',net_id,'train_iter_1000000');
    config_file = fullfile(net_dir,'2012','config',net_id,sprintf('deploy_%s_1.prototxt',layer));
    mean_file = fullfile(net_dir,'2012','protobuf/train_mean_pca.bproto');
    im_size = [226 226 3]; 
    top_dim = 4096; diff_dim = prod([11 11 3 96]);
  case 4
    net_id = 'epitonb30'; layer = 'fc8';
    model_file = fullfile(net_dir,'2012','model',net_id,'train_iter_700000');
    config_file = fullfile(net_dir,'2012','config',net_id,sprintf('deploy_%s.prototxt',layer));
    mean_file = []; %fullfile(net_dir,'2012','protobuf/train_mean_pca.bproto');
    im_size = [220 220 3]; 
    top_dim = 4096; %diff_dim = prod([11 11 3 96]);
  case 5
    net_id = 'epito2b30'; layer = 'fc8';
    model_file = fullfile(net_dir,'2012','model',net_id,'train_iter_680000');
    config_file = fullfile(net_dir,'2012','config',net_id,sprintf('deploy_%s.prototxt',layer));
    mean_file = []; %fullfile(net_dir,'2012','protobuf/train_mean_pca.bproto');
    im_size = [220 220 3]; 
    top_dim = 4096; %diff_dim = prod([11 11 3 96]);
  case 6
    net_id = 'overfeat2b30'; layer = 'fc8';
    model_file = fullfile(net_dir,'2012','model',net_id,'train_iter_880000');
    config_file = fullfile(net_dir,'2012','config',net_id,sprintf('deploy_%s.prototxt',layer));
    mean_file = []; %fullfile(net_dir,'2012','protobuf/train_mean_pca.bproto');
    im_size = [216 216 3]; 
    top_dim = 4096; %diff_dim = prod([11 11 3 96]);
  case 7
    net_id = 'overfeatnb30'; layer = 'fc8';
    model_file = fullfile(net_dir,'2012','model',net_id,'train_iter_690000');
    config_file = fullfile(net_dir,'2012','config',net_id,sprintf('deploy_%s.prototxt',layer));
    mean_file = []; %fullfile(net_dir,'2012','protobuf/train_mean_pca.bproto');
    im_size = [216 216 3]; 
    top_dim = 4096; %diff_dim = prod([11 11 3 96]);
  case 8
    net_id = 'topon2b30'; layer = 'fc8';
    model_file = fullfile(net_dir,'2012','model',net_id,'train_iter_540000');
    config_file = fullfile(net_dir,'2012','config',net_id,sprintf('deploy_%s.prototxt',layer));
    mean_file = []; %fullfile(net_dir,'2012','protobuf/train_mean_pca.bproto');
    im_size = [220 220 3]; 
    top_dim = 4096; %diff_dim = prod([11 11 3 96]);
end

% init caffe network
if isempty(mean_file)
  caffe('init', config_file, model_file);
else
  caffe('init', config_file, model_file, mean_file, im_size);
end

% specify which device to use
caffe('set_device',0);
caffe('set_mode_gpu');
% to also cover BWD computation
caffe('set_phase_train');

%% Read images and extract features
disp('Extract Deep Features');

feat_type = 'fwd'; % 'fwd' / 'fbwd'
top_diff = randn([1 1 top_dim], 'single'); % same top_diff for all images

sub_data = '101';
%sub_data = '256';
data_dir = fullfile('/rmt/data/visual/caltech',sub_data);
list = read_list(fullfile(data_dir, 'list.txt'));
num_images = length(list);

switch feat_type

  case 'fwd'
    feat_dim = top_dim;
    features = zeros([feat_dim num_images],'single');
    batch_size = 50;
    images = zeros([im_size batch_size],'single');
    num_batches = ceil(num_images/batch_size);
    tic;
    for b = 1:num_batches
      % Read a batch of images
      cur_batch_size = 0;
      for i = 1:batch_size
        im_seq = i + (b-1) * batch_size;
        if im_seq > num_images
          break;
        end
        cur_batch_size = cur_batch_size + 1;
        image = single(imread(fullfile(data_dir, list{im_seq})));
        image = imresize(image, im_size(1:2), 'bilinear');
        if size(image,3) == 1
          image = repmat(image,[1 1 im_size(3)]);
        end
        images(:,:,:,i) = image;
      end
      feature = caffe('forward', images(:,:,:,1:cur_batch_size));
      features(:, (b-1)*batch_size + (1:cur_batch_size)) = reshape( ...
        feature, [feat_dim cur_batch_size]);
      print_progress(1, 10, b, num_batches);
    end
   
  case 'fbwd'
    feat_dim = top_dim + diff_dim;
    features = zeros([feat_dim num_images],'single');    
    images = zeros([im_size 1],'single');
    for im_seq = 1:num_images
      image = single(imread(fullfile(data_dir, list{im_seq})));
      image = imresize(image, im_size(1:2), 'bilinear');
      if size(image,3) == 1
        image = repmat(image,[1 1 im_size(3)]);
      end
      images = image;
      [feature, diff] = caffe('forward_backward', images, top_diff, 1);
      features(:, im_seq) = vertcat( ...
        reshape(feature, [top_dim 1]), ...
        reshape(diff{1}, [diff_dim 1]));
      print_progress(1, 10, im_seq, num_images);
    end
   
end

mat_name = fullfile(sub_data,sprintf('features_%s_%s.mat',layer,feat_type));
save(mat_name,'features','list','layer','feat_type');

%% Classify
disp('Classification Setup');

addpath('/home/gpapan/code/ext/libsvm-3.17/matlab');
%addpath('/home/gpapan/software/patrec/libsvm/libsvm-3.17/matlab');

% sub_data = '101';
% layer = 'fc7';
% feat_type = 'fwd';

mat_name = fullfile(sub_data,sprintf('features_%s_%s.mat',layer,feat_type));
tmp = load(mat_name);
features = tmp.features;
list = tmp.list;

[feat_dim, num_images] = size(features);
label_str = cellfun(@(x)fileparts(x),list,'UniformOutput',false);
label_set = unique(label_str);
num_labels = length(label_set);
labels = zeros(num_images,1);
for i=1:num_images
  for l=1:num_labels
    if strcmp(label_str{i},label_set{l})
      labels(i) = l;
      break;
    end
  end
end

% Pick 30 images/class as positive, rest negative
num_train_per_class = 30;
train_set = false([num_images 1]);
for l=1:num_labels
  Ind = find(labels==l);
  P = randperm(length(Ind));
  if length(Ind)<num_train_per_class
    warning('Too many training set images for class %s.', label_set{l});
    Ind_train = Ind;
  else
    Ind_train = Ind(P(1:num_train_per_class));
  end
  train_set(Ind_train) = true;
end
test_set = ~train_set;

%% Train/test an SVM
disp('Train/Test SVM');

% Nets trained on ILSVRC-2010:
%kermap = @(x) x; % epit2 (2010) 101: 84.6318% (5149/6084)
%kermap = @(x) max(x,0); % epit2 (2010) 101: 86.3412% (5253/6084)
%kermap = @(x) vertcat(max(x,0), max(-x,0)); % epit2 (2010) 101: 84.5989% (5147/6084)

% Nets trained on ILSVRC-2012:
%kermap = @(x) max(x,0); % alexnet2 101: 87.5082% (5324/6084) / 87.7548% (5339/6084) (bug in mean norm code)
%kermap = @(x) max(x,0); % alexnet2 101: 87.9027% (5348/6084) / 87.311% (5312/6084) / 88.0999% (5360/6084) / 88.4122% (5379/6084) (corrected bug in mean norm code)
kermap = @(x) l2_norm(max(x,0), 1); % alexnet2 101: 88.5437% (5387/6084)
%kermap = @(x) max(l2_norm(x,1), 0); % alexnet2 101: 83.2183% (5063/6084)

%kermap = @(x) vertcat(max(x(1:top_dim,:),0), 0.01 * x(top_dim+(1:diff_dim),:)); % 1.0x: 31.2788% (1903/6084) / 0.01x: 87.0151% (5294/6084)
%kermap = @(x) max(x(1:top_dim,:),0); % 88.3794% (5377/6084)
%kermap = @(x) x(top_dim+(1:diff_dim),:); % 31.213% (1899/6084)

model = svmtrain(labels(train_set), double(kermap(features(:,train_set))'), '-s 0 -t 0 -c 1 -q');
[pred_label, accuracy, decision_values] = svmpredict(labels(test_set), double(kermap(features(:,test_set))'), model);

%%
conf_matrix = zeros(102);
true_labels = labels(test_set);
num_test = length(true_labels);
for i=1:num_test
  conf_matrix(true_labels(i), pred_label(i)) = conf_matrix(true_labels(i), pred_label(i)) + 1;
end

accu = zeros(102,1);
for l=1:102
  accu(l) = conf_matrix(l,l) / sum(conf_matrix(l,:));
end

fprintf(1,'Mean accuracy = %.2f\n', 100 * mean(accu));

% alexnet2: Accuracy = 88.215% (5367/6084) / Mean accuracy = 84.52
% epitonb30: Accuracy = 89.3984% (5439/6084) / Mean accuracy = 87.37
% epito2b30: Accuracy = 89.5464% (5448/6084) / Mean accuracy = 87.85
% overfeat2b30: Accuracy = 89.119% (5422/6084) / Mean accuracy = 87.31
% overfeatnb30: Accuracy = 87.426% (5319/6084) / Mean accuracy = 85.34
% topon2b30: Accuracy = 87.1959% (5305/6084) / Mean accuracy = 85.76
