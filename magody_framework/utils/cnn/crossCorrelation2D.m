function result = crossCorrelation2D(input_original, kernel_original, mode, correlation_shape, stride, as_convolution)
% Same behaviour of c = xcorr2(a,b) with mode='full'

input = input_original;

if nargin < 5
	stride = 1;
end

if nargin < 6
	as_convolution = false;
end

if as_convolution
    kernel = rot90(kernel_original, 2);
else
    kernel = kernel_original;
end


shape_kernel_matrix = size(kernel);


% array
result = zeros(correlation_shape);

if mode == "full"
    offset_rows = (shape_kernel_matrix(1) - 1);
    offset_cols = (shape_kernel_matrix(2) - 1);
    input = zeroPadding2D(input, [offset_rows, offset_cols]);
end

input_size = size(input);
row = 1;
lim_rows = input_size(1) - shape_kernel_matrix(1) + 1;
lim_cols = input_size(2) - shape_kernel_matrix(2) + 1;

index_result_row = 1;
while row <= lim_rows
        
    col = 1;
    index_result_col = 1;
    while col <= lim_cols
        input_slice = input(row:row+shape_kernel_matrix(1)-1, col:col+shape_kernel_matrix(2)-1);
        correlation_value = input_slice .* kernel;
        result(index_result_row, index_result_col) = sum(correlation_value(:));
        col = col + stride;
        index_result_col = index_result_col + 1;
    end
    index_result_row = index_result_row + 1;
    row = row + stride;
end
    
end

