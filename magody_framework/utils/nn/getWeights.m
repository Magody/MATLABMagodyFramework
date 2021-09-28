function W = getWeights(mean, sigma, shape, mode)
% Reference: https://towardsdatascience.com/all-ways-to-initialize-your-neural-network-16a585574b52


% generally the output size
neurons_i = shape(1);
% generally the input size
neurons_iminus1 = shape(2);

if mode == "xavier"
    % xavier: good for tanh, sigmoid and just a little with Relu
    W = normrnd(mean, sigma, neurons_i, neurons_iminus1)* sqrt(1/neurons_iminus1);
 
elseif mode == "kaiming"
    % kaiming good with ReluNonlinearities. ReLU changes the activations and the variance is halved, so we need to double the variance to get the original effect of Xavier
    W = normrnd(mean, sigma, neurons_i, neurons_iminus1) * sqrt(2/neurons_iminus1);

else
    % default xavier: good for tanh, sigmoid and a little with Relu
    W = normrnd(mean, sigma, neurons_i, neurons_iminus1) * sqrt(1/neurons_iminus1);
 
end



end

