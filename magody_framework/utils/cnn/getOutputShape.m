function output_shape = getOutputShape(W_height, W_width, shape_kernel_matrix, P, S, filters)
    %{
    [(W−K+2P)/S]+1.
    W is the input volume
    shape_kernel_matrix - K is the Kernel size
    P is the padding - commonly 0
    S is the stride
    %}
    dim_height = floor((W_height-shape_kernel_matrix(1)+2 * P)/S+1);
    dim_width = floor((W_width-shape_kernel_matrix(2)+2 * P)/S+1);
    output_shape = [dim_height, dim_width, filters];
end 