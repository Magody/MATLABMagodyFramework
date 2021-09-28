function output = reshapeCorrelation(correlation, channels, shape_matrix)

[num_filters, ~] = size(correlation);

output = zeros([shape_matrix(1:2), num_filters]);

for filter=1:num_filters
    slice_filter = correlation(filter, :);
    % output(:, :, filter) = transpose(reshape(slice_filter(:), shape_matrix));
    output(:, :, filter) = reshape(slice_filter(:), shape_matrix);

end


end