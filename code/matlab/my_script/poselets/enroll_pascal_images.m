function im = enroll_pascal_images(pascal_folder, annot_folder, format)
% "enroll" all the annotated images
%
    fdir = dir(fullfile(annot_folder, ['*.' format]));
   
    files = cell(1, length(fdir));        

    for i=1:length(fdir)
       files{i}=fdir(i).name; 
    end

    for i = 1 : length(fdir)
        eiim=strfind(files{i},'_');
        if isempty(eiim)
            error('%s is nonstandard xml annotation name.', files{i});
        end
        files{i} = files{i}(1:eiim(end)-1);
    end
        
    files = unique(files);

    img_path = fullfile(pascal_folder, 'JPEGImages');
    
    im.stem = files;
    im.dims = zeros(numel(files), 2, 'uint32');
    im.image_directory = img_path;
    
    for i = 1 : numel(files)
        img = imread(fullfile(img_path, [files{i}, '.jpg']));
        im.dims(i, :) = [size(img, 2), size(img, 1)];
    end

end

