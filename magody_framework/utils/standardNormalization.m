function normaliced = standardNormalization(matrix, mode)
% normalices 2D only

    if mode == "horizontal2D"
        matrix_mean = mean(matrix, 2);
        matrix_std = std(matrix, 1, 2);
        normaliced = (matrix - matrix_mean) ./ repmat(matrix_std, [1, size(matrix, 2)]);

    elseif mode == "all"
        matrix_mean = mean(mean(matrix));
        matrix_std = mean(std(matrix));
        normaliced = (matrix - matrix_mean) / matrix_std;
    end
end
