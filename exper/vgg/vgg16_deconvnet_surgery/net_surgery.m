%%% clear nets
caffe.reset_all();

%%% variables
use_gpu = 1;
gpu_id  = 0;

orig_prototxt = './init0.prototxt';
orig_weights  = './init0.caffemodel';

target_prototxt = './init1.prototxt';
target_weights  = './init1.caffemodel';

%%%

% Add caffe/matlab to you Matlab search PATH to use matcaffe
if exist('../../../code/matlab/+caffe', 'dir')
  addpath('../../../code/matlab');
else
  error('Please run this demo from caffe/matlab/demo');
end

% Set caffe mode
if use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

% load the pretrained model
phase = 'test';
net_model   = orig_prototxt;
net_weights = orig_weights;
trained_net = caffe.Net(net_model, net_weights, phase);

% load the model with desired net structure
phase = 'train';
net_model = target_prototxt;
want_net  = caffe.Net(net_model, phase);

%%%%%% copy and do net surgery

%%% copy parameters from all the convolutional layers and batch_norm layers
want_layer_types = {'Convolution', 'Deconvolution', 'BN', 'BatchNorm'};

% find ids for those layers that you will perform surgery
surgery_layer_name = {'fc6', 'fc7', 'fc6-deconv'};
surgery_layer_ids = -1 * ones(1, numel(surgery_layer_name));

for k = 1 : numel(surgery_layer_name)
    surgery_layer_ids(k) = trained_net.name2layer_index(...
        surgery_layer_name{k});
end

for layer = 1 : numel(trained_net.layer_vec)
    layer_type = trained_net.layer_vec(layer).type;
   
    if layer == surgery_layer_ids(1) 
        name_id = 1;
        % fc6
        param1 = trained_net.layer_vec(surgery_layer_ids(name_id)).params(1).get_data();
        param2 = trained_net.layer_vec(surgery_layer_ids(name_id)).params(2).get_data();
        
        % 7x7x512x4096 -> 7x7x512x1024
        param1 = param1(:, :, :, 1:4:end);
        param2 = param2(1:4:end);
        
        want_net.layer_vec(surgery_layer_ids(name_id)).params(1).set_data(param1);
        want_net.layer_vec(surgery_layer_ids(name_id)).params(2).set_data(param2);
    elseif layer == surgery_layer_ids(1) + 1
        name_id = 1;
        % bn right after fc6
        param1 = trained_net.layer_vec(surgery_layer_ids(name_id) + 1).params(1).get_data();
        param2 = trained_net.layer_vec(surgery_layer_ids(name_id) + 1).params(2).get_data();
        
        % 1x1x4096 -> 1x1x1024
        param1 = param1(1:4:end);
        param2 = param2(1:4:end);
        
        want_net.layer_vec(surgery_layer_ids(name_id) + 1).params(1).set_data(param1);
        want_net.layer_vec(surgery_layer_ids(name_id) + 1).params(2).set_data(param2);
    elseif layer == surgery_layer_ids(2)
        name_id = 2;
        % fc7
        param1 = trained_net.layer_vec(surgery_layer_ids(name_id)).params(1).get_data();
        param2 = trained_net.layer_vec(surgery_layer_ids(name_id)).params(2).get_data();
        
        % 1x1xx4096x4096 -> 1x1x1024x1024
        param1 = param1(:, :, 1:4:end, 1:4:end);
        param2 = param2(1:4:end);
        
        want_net.layer_vec(surgery_layer_ids(name_id)).params(1).set_data(param1);
        want_net.layer_vec(surgery_layer_ids(name_id)).params(2).set_data(param2);
    elseif layer == surgery_layer_ids(2) + 1
        name_id = 2;
        % bn right after fc7
        param1 = trained_net.layer_vec(surgery_layer_ids(name_id) + 1).params(1).get_data();
        param2 = trained_net.layer_vec(surgery_layer_ids(name_id) + 1).params(2).get_data();
        
        % 1x1x4096 -> 1x1x1024
        param1 = param1(1:4:end);
        param2 = param2(1:4:end);
        
        want_net.layer_vec(surgery_layer_ids(name_id) + 1).params(1).set_data(param1);
        want_net.layer_vec(surgery_layer_ids(name_id) + 1).params(2).set_data(param2);   
    elseif layer == surgery_layer_ids(3)
        name_id = 3;
        % fc6-deconv
        param1 = trained_net.layer_vec(surgery_layer_ids(name_id)).params(1).get_data();
        param2 = trained_net.layer_vec(surgery_layer_ids(name_id)).params(2).get_data();
        
        % 7x7x512x4096 -> 7x7x512x1024
        param1 = param1(:, :, :, 1:4:end);
        % no need to do surgery on param2
        
        want_net.layer_vec(surgery_layer_ids(name_id)).params(1).set_data(param1);
        want_net.layer_vec(surgery_layer_ids(name_id)).params(2).set_data(param2);
        
%     % no need to do surgery on the bn right right fc6-deconv
%     elseif layer == surgery_layer_ids(3) + 1
%         name_id = 3;
%         % bn right after fc6-deconv
%         param1 = trained_net.layer_vec(surgery_layer_ids(name_id) + 1).params(1).get_data();
%         param2 = trained_net.layer_vec(surgery_layer_ids(name_id) + 1).params(2).get_data();
%         
%         %1x1x4096 -> 1x1x1024
%         param1 = param1(1:4:end);
%         param2 = param2(1:4:end);
%         
%         want_net.layer_vec(surgery_layer_ids(name_id) + 1).params(1).set_data(param1);
%         want_net.layer_vec(surgery_layer_ids(name_id) + 1).params(2).set_data(param2); 
    elseif ismember(layer_type, want_layer_types)
        % copy first two parameters
        % conv/deconv: weight and bias
        % bn: scale and shift
        for p = 1 : 2 
            want_net.layer_vec(layer).params(p).set_data(...
                trained_net.layer_vec(layer).params(p).get_data());
        end
    end
end

%%%%% save results
want_net.save(target_weights);

%%% clear nets
caffe.reset_all();



















