function zero_padded = zeroPadding2D(X, shape_pad)

    shape_input = size(X);
    zero_padded = zeros([shape_input(1) + 2 * shape_pad(1), shape_input(2) + 2 * shape_pad(2)]);
    
    row_begin = shape_pad(1) + 1;
    row_end = row_begin + shape_input(1) - 1;
    col_begin = shape_pad(2) + 1;
    col_end = col_begin + shape_input(2) - 1;
    
    zero_padded(row_begin:row_end, col_begin:col_end) = X;
end