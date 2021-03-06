
function vectorized = vectorizeKernels(kernels)
    % kernel is 4-D matrix, [height, width, input_depth, num_filters]

    [rows, cols, input_depth, num_filters] = size(kernels);
    
    elements_per_kernel = rows * cols;
    vectorized = zeros([num_filters, elements_per_kernel * input_depth]);



    for filter=1:num_filters
        for depth=1:input_depth
            kernel = kernels(:, :, depth, filter);
            
            index_depth_begin = (depth-1) * elements_per_kernel + 1;
            index_depth_end = depth * elements_per_kernel;
            vectorized(filter, index_depth_begin:index_depth_end) = kernel(:);
            
        end
        
        
    end

end