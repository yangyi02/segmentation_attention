function pimap = my_part2ind()
% Define the part index of each objects. 
% One can merge different parts by using the same index for the
% parts that are desired to be merged. 
% For example, one can merge 
% the left lower leg (llleg) and the left upper leg (luleg) of person by setting: 
% pimap{15}('llleg')      = 19;               % left lower leg
% pimap{15}('luleg')      = 19;               % left upper leg

pimap = cell(20, 1);                    
% Will define part index map for the 20 PASCAL VOC object classes in ascending
% alphabetical order (the standard PASCAL VOC order). 
for ii = 1:20
    pimap{ii} = containers.Map('KeyType','char','ValueType','int32');
end

% [aeroplane]
pimap{1}('body')        = 0;                
pimap{1}('stern')       = 0;                                      
pimap{1}('lwing')       = 0;                % left wing
pimap{1}('rwing')       = 0;                % right wing
pimap{1}('tail')        = 0;                
for ii = 1:10
    pimap{1}(sprintf('engine_%d', ii)) = 0; % multiple engines
end
for ii = 1:10
    pimap{1}(sprintf('wheel_%d', ii)) = 0;  % multiple wheels
end

% [bicycle]
pimap{2}('fwheel')      = 0;                % front wheel
pimap{2}('bwheel')      = 0;                % back wheel
pimap{2}('saddle')      = 0;               
pimap{2}('handlebar')   = 0;                % handle bar
pimap{2}('chainwheel')  = 0;                % chain wheel
for ii = 1:10
    pimap{2}(sprintf('headlight_%d', ii)) = 0;
end

% [bird]
pimap{3}('head')        = 0;
pimap{3}('leye')        = 0;                % left eye
pimap{3}('reye')        = 0;                % right eye
pimap{3}('beak')        = 0;               
pimap{3}('torso')       = 0;            
pimap{3}('neck')        = 0;
pimap{3}('lwing')       = 0;                % left wing
pimap{3}('rwing')       = 0;                % right wing
pimap{3}('lleg')        = 0;                % left leg
pimap{3}('lfoot')       = 0;               % left foot
pimap{3}('rleg')        = 0;               % right leg
pimap{3}('rfoot')       = 0;               % right foot
pimap{3}('tail')        = 0;

% [boat]
% only has silhouette mask 

% [bottle]
pimap{5}('cap')         = 0;
pimap{5}('body')        = 0;

% [bus]
pimap{6}('frontside')   = 0;
pimap{6}('leftside')    = 0;
pimap{6}('rightside')   = 0;
pimap{6}('backside')    = 0;
pimap{6}('roofside')    = 0;
pimap{6}('leftmirror')  = 0;
pimap{6}('rightmirror') = 0;
pimap{6}('fliplate')    = 0;                % front license plate
pimap{6}('bliplate')    = 0;                % back license plate
for ii = 1:10
    pimap{6}(sprintf('door_%d',ii)) = 0;
end
for ii = 1:10
    pimap{6}(sprintf('wheel_%d',ii)) = 0;
end
for ii = 1:10
    pimap{6}(sprintf('headlight_%d',ii)) = 0;
end
for ii = 1:20
    pimap{6}(sprintf('window_%d',ii)) = 0;
end

% [car]
keySet = keys(pimap{6});
valueSet = values(pimap{6});
pimap{7} = containers.Map(keySet, valueSet);         % car has the same set of parts with bus

% [cat]
pimap{8}('head')        = 0;
pimap{8}('leye')        = 0;                % left eye
pimap{8}('reye')        = 0;                % right eye
pimap{8}('lear')        = 0;                % left ear
pimap{8}('rear')        = 0;                % right ear
pimap{8}('nose')        = 0;
pimap{8}('torso')       = 0;   
pimap{8}('neck')        = 0;
pimap{8}('lfleg')       = 0;                % left front leg
pimap{8}('lfpa')        = 0;               % left front paw
pimap{8}('rfleg')       = 0;               % right front leg
pimap{8}('rfpa')        = 0;               % right front paw
pimap{8}('lbleg')       = 0;               % left back leg
pimap{8}('lbpa')        = 0;               % left back paw
pimap{8}('rbleg')       = 0;               % right back leg
pimap{8}('rbpa')        = 0;               % right back paw
pimap{8}('tail')        = 0;               

% [chair]
% only has sihouette mask 

% [cow]
pimap{10}('head')       = 0;
pimap{10}('leye')       = 0;                % left eye
pimap{10}('reye')       = 0;                % right eye
pimap{10}('lear')       = 0;                % left ear
pimap{10}('rear')       = 0;                % right ear
pimap{10}('muzzle')     = 0;
pimap{10}('lhorn')      = 0;                % left horn
pimap{10}('rhorn')      = 0;                % right horn
pimap{10}('torso')      = 0;            
pimap{10}('neck')       = 0;
pimap{10}('lfuleg')     = 0;               % left front upper leg
pimap{10}('lflleg')     = 0;               % left front lower leg
pimap{10}('rfuleg')     = 0;               % right front upper leg
pimap{10}('rflleg')     = 0;               % right front lower leg
pimap{10}('lbuleg')     = 0;               % left back upper leg
pimap{10}('lblleg')     = 0;               % left back lower leg
pimap{10}('rbuleg')     = 0;               % right back upper leg
pimap{10}('rblleg')     = 0;               % right back lower leg
pimap{10}('tail')       = 0;               

% [diningtable]
% only has silhouette mask 

% [dog]
keySet = keys(pimap{8});
valueSet = values(pimap{8});
pimap{12} = containers.Map(keySet, valueSet);         	% dog has the same set of parts with cat, 
                                            		% except for the additional
                                            		% muzzle
pimap{12}('muzzle')     = 0;

% [horse]
keySet = keys(pimap{10});
valueSet = values(pimap{10});
pimap{13} = containers.Map(keySet, valueSet);        	% horse has the same set of parts with cow, 
                                                        % except it has hoof instead of horn
remove(pimap{13}, {'lhorn', 'rhorn'});
pimap{13}('lfho') = 0;
pimap{13}('rfho') = 0;
pimap{13}('lbho') = 0;
pimap{13}('rbho') = 0;

% [motorbike]
pimap{14}('fwheel')     = 0;
pimap{14}('bwheel')     = 0;
pimap{14}('handlebar')  = 0;
pimap{14}('saddle')     = 0;
for ii = 1:10
    pimap{14}(sprintf('headlight_%d', ii)) = 0;
end

% [person]
pimap{15}('head')       = 1;
pimap{15}('leye')       = 1;                    % left eye
pimap{15}('reye')       = 1;                    % right eye
pimap{15}('lear')       = 1;                    % left ear
pimap{15}('rear')       = 1;                    % right ear
pimap{15}('lebrow')     = 1;                    % left eyebrow    
pimap{15}('rebrow')     = 1;                    % right eyebrow
pimap{15}('nose')       = 1;                    
pimap{15}('mouth')      = 1;                    
pimap{15}('hair')       = 1;                   

pimap{15}('torso')      = 2;                   
pimap{15}('neck')       = 1;           
pimap{15}('llarm')      = 4;                   % left lower arm
pimap{15}('luarm')      = 3;                   % left upper arm
pimap{15}('lhand')      = 4;                   % left hand
pimap{15}('rlarm')      = 4;                   % right lower arm
pimap{15}('ruarm')      = 3;                   % right upper arm
pimap{15}('rhand')      = 4;                   % right hand

pimap{15}('llleg')      = 6;               	% left lower leg
pimap{15}('luleg')      = 5;               	% left upper leg
pimap{15}('lfoot')      = 6;               	% left foot
pimap{15}('rlleg')      = 6;               	% right lower leg
pimap{15}('ruleg')      = 5;               	% right upper leg
pimap{15}('rfoot')      = 6;               	% right foot

% [pottedplant]
pimap{16}('pot')        = 0;
pimap{16}('plant')      = 0;

% [sheep]
keySet = keys(pimap{10});
valueSet = values(pimap{10});
pimap{17} = containers.Map(keySet, valueSet);        % sheep has the same set of parts with cow

% [sofa]
% only has sihouette mask 

% [train]
pimap{19}('head')       = 0;
pimap{19}('hfrontside') = 0;                	% head front side                
pimap{19}('hleftside')  = 0;                	% head left side
pimap{19}('hrightside') = 0;                	% head right side
pimap{19}('hbackside')  = 0;                 	% head back side
pimap{19}('hroofside')  = 0;                	% head roof side

for ii = 1:10
    pimap{19}(sprintf('headlight_%d',ii)) = 0;
end

for ii = 1:10
    pimap{19}(sprintf('coach_%d',ii)) = 0;
end

for ii = 1:10
    pimap{19}(sprintf('cfrontside_%d', ii)) = 0;   % coach front side
end

for ii = 1:10
    pimap{19}(sprintf('cleftside_%d', ii)) = 0;   % coach left side
end

for ii = 1:10
    pimap{19}(sprintf('crightside_%d', ii)) = 0;  % coach right side
end

for ii = 1:10
    pimap{19}(sprintf('cbackside_%d', ii)) = 0;   % coach back side
end

for ii = 1:10
    pimap{19}(sprintf('croofside_%d', ii)) = 0;   % coach roof side
end


% [tvmonitor]
pimap{20}('screen')     = 0;

