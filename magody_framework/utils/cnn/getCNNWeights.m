function W = getCNNWeights(mean, sigma, shape, previous_neurons)

W = normrnd(mean, sigma, shape)/sqrt(previous_neurons);
 
end

