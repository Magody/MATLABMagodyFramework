%% Clean up
clear all;
clc;
close all;

seed_rng = 44;
%% Libs
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
dir_base_models_result = "/home/magody/programming/MATLAB/deep_learning_from_scratch/examples/convolutional_neural_network/models_result";

addpath(genpath(path_to_framework));

%% mock data
lim = 24;
X = standardNormalization([1:2:lim; 2:2:lim; 2:2:lim; 1:2:lim], ...
                          "all");
y = [[1, 0]; [0, 1]; [0, 1]; [1, 0]];


%% setup
generate_rng(seed_rng);
epochs = 100;
learning_rate = 0.01;
decay_rate_alpha = 0.01;
batch_size = 64;

shape_input = [1, size(X, 2), 1];
conv_sequential = Sequential({...
    Convolutional([1, 3], 1, 0, 1, shape_input), ...
    Activation("relu"), ....
    Pooling("mean", [1, 2]), ...
    Reshape(), ...
});
input_dense = prod(conv_sequential.shape_output);
input_dense = size(X, 2);
sequential = Sequential({...
    Dense(4, input_dense), ...
    Activation("relu"), ...
    Dense(2), ...
    ActivationOnlyForward("softmax"), ...
});

%% train
m = size(X, 1);
history_errors = zeros([1, epochs]);
for epoch=1:epochs
    error = 0;
    for sample=1:m
        x = X(sample, :);
        % x = conv_sequential.forward();
        test_forward = sequential.forward(x(:));
        error = error + Loss.binary_cross_entropy(y(sample, :)', test_forward);
        
        grad = Loss.softmaxGradient(y(sample, :)', test_forward);
        grad = sequential.backward(grad, learning_rate);
        % grad_conv = conv_sequential.backward(grad, learning_rate);
    end
    history_errors(1, epoch) = error;
     
end
plot(history_errors);