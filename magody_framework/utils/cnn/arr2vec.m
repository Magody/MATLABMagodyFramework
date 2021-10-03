function vec = arr2vec(x, kernel_size)

    stride = 1;
    padding = 0;
    k1 = kernel_size(1);
    k2 = kernel_size(2);
    
    [n1, n2, ch, mb] = size(x);
        
    y = zeros([n1+2*padding, n2+2*padding, ch, mb]);
    
    y(padding+1:n1+padding,padding+1:n2+padding, :, :) = x;
    
    limit_i = n1-k1+2*padding+1;
    limit_j = n2-k2+2*padding+1;
    start_idx = zeros([1, limit_i*limit_j]);
    
    index_start_idx = 1;
    for i=0:stride:limit_i-1
        for j=0:stride:limit_j-1
            start_idx(1, index_start_idx) = j + (n2+2*padding)* i;
            index_start_idx = index_start_idx + 1;
        end
    end
    
    index_grid = 1;
    grid = zeros([1, ch*k1*k2]);
    for k=0:ch-1 
        for i=0:k1-1
            for j=0:k2-1
                grid(1, index_grid) = j + (n2+2*padding)*i + (n1+2*padding) * (n2+2*padding) * k;
                index_grid = index_grid + 1;
            end
        end
    end
    
    
    % create as much rows as values in start_idx
    % then sum each value of start idx to each corresponding row
    to_take = start_idx(:) + grid;  % horizontal + vertical = rectangular
    
    init_range = 0:(mb-1);
    batch = init_range * ch * (n1+2*padding) * (n2+2*padding);
    y_take = zeros([size(to_take), length(batch)]);
    
    for index_b=1:size(y_take, 3)
        y_take(:, :, index_b) = to_take + batch(index_b) ;
    end
    
    y = y(:);
    vec = zeros(size(y_take));
    
    for index_b=1:size(y_take, 3)
        for index_row=1:size(y_take, 1)
            for index_col=1:size(y_take, 2)
                vec(index_row, index_col, index_b) = y(y_take(index_row, index_col, index_b) + 1); % +1 for matlab indexes
            end
        end
    end
            
            
    
end