function [normaliced, data_mean, data_std] = standardization(matrix, mode)
% normalices 2D only

    if mode == "horizontal2D"
        data_mean = mean(matrix, 2);
        data_std = std(matrix, 1, 2);
        normaliced = (matrix - data_mean) ./ repmat(data_std, [1, size(matrix, 2)]);
    elseif mode == "vertical2D"
        data_mean = mean(matrix, 1);
        data_std = std(matrix, 1, 1);
        normaliced = (matrix - data_mean) ./ repmat(data_std, [size(matrix, 1), 1]);

    elseif mode == "all"
        data_mean = mean(matrix(:));
        data_std = std(matrix(:));
        normaliced = (matrix - data_mean) / data_std;
    end
end
