function output = reshapeCorrelation(correlation, channels, shape_matrix)

[num_filters, ~] = size(correlation);

output = zeros([shape_matrix(1:2), num_filters]);

for filter=1:num_filters
    
    output(:, :, filter) = transpose(reshape(correlation(filter, :), shape_matrix));

end


end