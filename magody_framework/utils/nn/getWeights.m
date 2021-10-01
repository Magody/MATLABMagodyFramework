function W = getWeights(mean, sigma, shape, previous_neurons, mode)
% Reference: https://towardsdatascience.com/all-ways-to-initialize-your-neural-network-16a585574b52


if mode == "xavier"
    % xavier: good for tanh, sigmoid and just a little with Relu
    W = normrnd(mean, sigma, shape)* sqrt(1/previous_neurons);
 
elseif mode == "kaiming" || mode == "He"
    % also called He, Is Kaiming He
    % kaiming good with ReluNonlinearities. ReLU changes the activations and the variance is halved, so we need to double the variance to get the original effect of Xavier
    % (1+a.^2) * previous_neurons), a in RelU is 0
    a = 0;  % other relu should change it
    W = normrnd(mean, sigma, shape) * sqrt(2/((1+a.^2) * previous_neurons));

else
    % default xavier: good for tanh, sigmoid and a little with Relu
    W = normrnd(mean, sigma, shape) * sqrt(1/previous_neurons);
 
end



end

