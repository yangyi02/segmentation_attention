save_fn = 'keypoints.mat';

pascal_folder = '~/workspace/rmt/data/pascal/VOCdevkit/VOC2012';
root_folder = '../../annotations/person';

im_fn = 'im_info.mat';

global im;
global config;

if ~exist(im_fn, 'file')
    im = enroll_pascal_images(pascal_folder, root_folder, 'xml');
    save(im_fn, 'im');
else
    tmp = load(im_fn);
    im = tmp.im;
    clear tmp;
end

config = init;
annots = read_annotations(root_folder);

save(save_fn, 'annots');