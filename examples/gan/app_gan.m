%% Clean up
clear all;
clc;
close all;
seed_rng = 44;

%% Libs
%{
setenv("path_to_framework", "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework");
setenv("path_to_mnist_cloth_dataset", "/home/magody/programming/MATLAB/deep_learning_from_scratch/examples/dataset");
%} 
path_to_framework = getenv("path_to_framework");
path_to_mnist_cloth_dataset = getenv("path_to_mnist_cloth_dataset");

addpath(genpath(path_to_framework));
addpath(genpath(path_to_mnist_cloth_dataset));


%% Generator

epochs = 10;
learning_rate = 0.003;
decay_rate_alpha = 0.1;
batch_size = 64;
dim_noise = 3000;

dim_signal = 996;

sequential = Sequential({
    Dense(2000, 'kaiming', dim_noise), ...
    Activation("relu"), ...
    Dense(dim_signal, 'xavier')
});


nnConfig = NNConfig(epochs, learning_rate, batch_size, "softmax_cross_entropy");
nnConfig.decay_rate_alpha = decay_rate_alpha;

generator = NeuralNetwork(sequential, nnConfig);

%% Discriminator

epochs = 10;
learning_rate = 0.003;
decay_rate_alpha = 0.1;
batch_size = 64;

discriminator_sequential = Sequential({
    Dense(128, 'kaiming', dim_signal), ...
    Activation("relu"), ...
    Dense(64, 'kaiming'), ...
    Activation("relu"), ...
    Dense(2), ...
    ActivationOnlyForward("softmax")
});


nnConfig = NNConfig(epochs, learning_rate, batch_size, "softmax_cross_entropy");
nnConfig.decay_rate_alpha = decay_rate_alpha;

discriminator = NeuralNetwork(discriminator_sequential, nnConfig);

%% train
data = load('emg.mat');
real_data = repmat(data.emg(:, 1), [1, 100]);

context = containers.Map(["is_test"], ["false"]);

noise = rand([dim_noise, batch_size]);

generated_signals = generator.forwardFull(noise, context);
output_for_real = discriminator.forwardFull(real_data(:, 1:32), context);
output_for_fake = discriminator.forwardFull(generated_signals(:, 1:32), context);
output = [output_for_real, output_for_fake];
y = [repmat([0; 1], [1, 32]), repmat([1; 0], [1, 32])];

discriminator.backward(y, output)
X_train = [generated_signals, real_data(:, 1)];
y_train = [0 1; 0 1];

generator.backward([1], output_for_fake);

%% prediction
fprintf("\nPREDICTION\n");
raw_y_pred = neural_network.predict(X_test);
[~, idx_pred] = max(raw_y_pred);
[~, idx_real] = max(transpose(y_test));

accuracy = sum(idx_pred == idx_real)/length(idx_pred);
fprintf("Accuracy: %.4f\n", accuracy);
