
%dataset = 'cloth_data_6';
dataset = 'voc10_part';


if strcmp(dataset, 'voc10_part')
    %res_folder = './new_res_part';
    res_folder = './res_part';

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
        
%         '1/007513778'   
%         '1/008023375'    
%         '2/000638587'
%         '2/001212077'
%         '2/002440078'
%         '2/002619671'
%         '2/003934354'
%         '2/003942681'
        };
    
    video_folder = './videos_part';
    frame_rate = 6;
elseif strcmp(dataset, 'cloth_data_6')
    res_folder = './res_cloth';
    img_folders = {'1'};
    
    video_folder = './videos_cloth';
    frame_rate = 1;
end


for ii = 1 : numel(img_folders)
    img_folder = img_folders{ii};
    
    folder_name = strrep(img_folder, '/', '_');
    img_dir = dir(fullfile(res_folder, folder_name, '*.jpg'));
    
    save_folder = fullfile(video_folder, folder_name);
    if ~exist(save_folder, 'dir')
        mkdir(save_folder)
    end
        
% %     % change file name
% %     for jj = 1 : numel(img_dir)
% %         img = imread(fullfile(res_folder, folder_name, img_dir(jj).name));
% %         save_frame_folder = fullfile(new_res_folder, folder_name);
% %         if ~exist(save_frame_folder, 'dir')
% %             mkdir(save_frame_folder)
% %         end
% %         img_fn = img_dir(jj).name(1:end-4);
% %         imwrite(img, fullfile(save_frame_folder, sprintf('0%s.jpg', img_fn)));
% %     end
    
    
    fn = fullfile(save_folder, [folder_name, '.avi']);
    writerObj = VideoWriter(fn);
    writerObj.FrameRate = frame_rate;
    open(writerObj);
    
    for jj = 1 : numel(img_dir)
        img = imread(fullfile(res_folder, folder_name, img_dir(jj).name));
        writeVideo(writerObj, img);
    end
    close(writerObj);
end
        