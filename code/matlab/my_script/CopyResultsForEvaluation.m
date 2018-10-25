function CopyResultsForEvaluation(from_folder, to_folder, VOCopts, id, task)
% copy the files that will be evaluated from from_folder to to_folder
%
% get the list from the from_folder, and only save those to to_folder that will be evaluated
if strcmp(task, 'seg')
  [gtids, gt]=textread(sprintf(VOCopts.seg.imgsetpath,VOCopts.testset),'%s %d');
  for i = 1 : numel(gtids)
    if mod((i+1), 100) == 0
      fprintf(1, 'seg processing %d (%d)...\n', i, numel(gtids));
    end

    from_fn = fullfile(from_folder, [gtids{i} '.png']);
    to_fn = fullfile(to_folder, [gtids{i} '.png']);

    if exist(from_fn, 'file')
      copyfile(from_fn, to_fn);
    end
  end
elseif strcmp(task, 'cls')
  for k = 1 : numel(VOCopts.classes)
    fprintf(1, 'cls processing class %s...\n', VOCopts.classes{k});

    cls = VOCopts.classes{k};  

    [gtids, gt] = textread(sprintf(VOCopts.clsimgsetpath, cls, VOCopts.testset),'%s %d');

    fn = [from_folder sprintf('/%s_cls_%s_%s.txt', id, VOCopts.testset, cls)];

    [ids,confidence] = textread(fn,'%s %f');
    [eval_list, ind] = intersect(ids, gtids);
    eval_conf = confidence(ind);

    if numel(eval_list) ~= numel(gtids)
      fprintf(1, 'WARNING: Eval_list (%d) and Gt_list (%d) not match...%d missing images\n', numel(eval_list), numel(gtids), abs(numel(gtids)-numel(eval_list)));
    end

    cls_res_fn = sprintf('%s_cls_%s_%s.txt', id, VOCopts.testset, cls);
    cls_fid = fopen(fullfile(to_folder, cls_res_fn), 'w');

    for kk = 1 : numel(eval_list)    
      fprintf(cls_fid, '%s %f\n', eval_list{kk}, eval_conf(kk));
    end
       
    fclose(cls_fid);
  end
else
  error('Wrong task\n');
end

