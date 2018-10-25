function [rec,prec,ap] = VOCevalcls(VOCopts, id)

num_class = numel(VOCopts.classes);

rec  = cell(1, num_class);
prec = cell(1, num_class);
ap   = zeros(1, num_class);

for k = 1 : num_class
  cls = VOCopts.classes{k};
  
  % load test set
  [gtids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');

  % hash image ids
  hash=VOChash_init(gtids);

  % load results
  [ids,confidence]=textread(sprintf(VOCopts.clsrespath,id,cls),'%s %f');

  % map results to ground truth images
  out=ones(size(gt))*-inf;
  tic;
  for i=1:length(ids)
      % display progress
      if toc>1
          fprintf('%s: pr: %d/%d\n',cls,i,length(ids));
          drawnow;
          tic;
      end
    
      % find ground truth image
      j=VOChash_lookup(hash,ids{i});
      if isempty(j)
          error('unrecognized image "%s"',ids{i});
      elseif length(j)>1
          error('multiple image "%s"',ids{i});
      else
          out(j)=confidence(i);
      end
  end
  
  % check if there are any missing images
  finite_ind = isfinite(out);
  finite_out = out(finite_ind);

  if numel(finite_out) < numel(gtids)
    fprintf(1, 'there are %d missing images...\n', numel(gtids) - numel(finite_out));
  end

  finite_gt = gt(finite_ind);

  % compute precision/recall
  [so,si]=sort(-finite_out);
  tp=finite_gt(si)>0;
  fp=finite_gt(si)<0;

  %[so,si]=sort(-out);
  %tp=gt(si)>0;
  %fp=gt(si)<0;

  fp=cumsum(fp);
  tp=cumsum(tp);
  
  rec{k} = tp / sum(finite_gt > 0);
  prec{k} = tp ./ (fp + tp);

  %rec{k}=tp/sum(gt>0);
  %prec{k}=tp./(fp+tp);

  ap(k)=VOCap(rec{k},prec{k});
 
  fprintf(1, 'class: %s, subset: %s, AP = %.3f\n',cls,VOCopts.testset,ap(k));
end

fprintf(1, 'mean AP = %.3f\n', mean(ap));
