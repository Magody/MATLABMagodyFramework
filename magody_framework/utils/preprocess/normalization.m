function [normaliced, data_min, data_max] = normalization(matrix, mode)
% normalices 2D only

    if mode == "horizontal2D"
        data_min = min(matrix, [], 2);
        data_max = max(matrix, [], 2);
        normaliced = (matrix - data_min) ./ repmat(data_max, [1, size(matrix, 2)]);
    elseif mode == "vertical2D"
        data_min = mean(matrix, [], 1);
        data_max = max(matrix, [], 1);
        normaliced = (matrix - data_min) ./ repmat(data_max, [size(matrix, 1), 1]);

    elseif mode == "all"
        data_min = min(matrix(:));
        data_max = max(matrix(:));
        normaliced = (matrix - data_min) / data_max;
    end
end
