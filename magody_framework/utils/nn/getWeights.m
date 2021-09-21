function W = getWeights(mean, sigma, shape)

% generally the output size
neurons_i = shape(1);
% generally the input size
neurons_iminus1 = shape(2);

W = normrnd(mean, sigma, neurons_i, neurons_iminus1)/sqrt(neurons_iminus1);
 
end

