function arr = vec2arr(x, kernel_size, old_shape)

    stride = 1;
    padding = 0;
    
    k1 = kernel_size(1);
    k2 = kernel_size(2);
    
    n = old_shape(1);
    p = old_shape(2);
    
    
    [md, ftrs, mb] = size(x);
    ch = floor(ftrs / (k1*k2));
    
    
    idx = zeros([k1 * k2, 2, n * p]);
    
    for i=1:n 
        for j=1:p
            
            index_i_j = (i - 1) * p + j;
            for k1i=1:k1 
                for k2j=1:k2
                    
                    index_k1_k2 = (k1i-1)*k2 + k2j;
                    
                    idx(index_k1_k2, 1, index_i_j) = (i-1)-(k1i-1);
                    idx(index_k1_k2, 2, index_i_j) = (j-1)-(k2j-1);
                end
            end
        end        
    end
    
    idx_segment1 = transpose(squeeze(idx(:,1,:)));
    idx_segment2 = transpose(squeeze(idx(:,2,:)));
    
    
    in_bounds = (idx_segment1 >= -padding) .* (idx_segment1 <= n-k1+padding);
    in_bounds = in_bounds .* (idx_segment2 >= -padding) .* (idx_segment2 <= p-k2+padding);
    
    in_strides = mod(idx_segment1+padding,stride) == 0 == mod(idx_segment2+padding, stride)==0;
    
    
    size_ch_concat = size(idx_segment1);
    to_take = zeros([size_ch_concat(1)*ch, size_ch_concat(2)]);
    for c=0:ch-1
        % +1 for matlab
        index_begin = size_ch_concat(1)*c;
        index_end = index_begin + size_ch_concat(1);
        to_take((index_begin+1):(index_end), :) = idx_segment1 * k2 + idx_segment2 + k1*k2*c;
    end
    
    filters_matrix = zeros([1, k1*k2]);
    for i=0:(k1*k2-1)
        filters_matrix(1, i+1) = ftrs * i;
    end
    to_take = to_take + filters_matrix;
    
    
    to_take_mb = zeros([size(to_take, 1)*mb, size(to_take, 2)]);
    for m=0:mb-1
        % +1 for matlab
        index_begin = size(to_take, 1)*m;
        index_end = index_begin + size(to_take, 1);
        to_take_mb((index_begin+1):index_end, :) = to_take + md*ftrs*m;
    end
    
    to_take = to_take_mb;
    
    in_bounds = repmat(in_bounds .* in_strides, [ch * mb,1]);
    
    
    x = x(:);
    x_take = zeros(size(to_take));
    
    
    for index_row=1:size(x_take, 1)
        for index_col=1:size(x_take, 2)
            x_take(index_row, index_col) = x(to_take(index_row, index_col)+1); % +1 for matlab indexes
        end
    end
    
    bounds_select = zeros(size(in_bounds));
    
    for i=1:size(bounds_select, 1)
        for j=1:size(bounds_select, 2)
            if in_bounds(i, j)
                bounds_select(i, j) = x_take(i, j);
            end
        end
    end
    
    
    
    arr = pagetranspose(reshape(sum(bounds_select, 2), [n, p, ch, mb]));
end