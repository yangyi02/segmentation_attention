function [out, row, col] = PadAndScaleImage(in, sc_size, input_max_size)
% scale the input image to sc_size
% And then pad it to input_max_size with zeros
%

if sc_size ~= 1
  tmp = imresize(in, sc_size, 'bilinear');
else
  tmp = in;
end

[row, col, channel] = size(tmp);
assert(row <= input_max_size && col <= input_max_size);

out = zeros(input_max_size, input_max_size, channel);
out(1:row, 1:col, 1:channel) = tmp;
