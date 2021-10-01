%% Clean up
clear all;
clc;
close all;
seed_rng = 44;
generate_rng(seed_rng);
a = ceil(rand * 10000);
assert(a == 5680);

%% Libs
%{
setenv("path_to_framework", "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework");
setenv("path_to_mnist_cloth_dataset", "/home/magody/programming/MATLAB/deep_learning_from_scratch/examples/dataset");
%} 
path_to_framework = getenv("path_to_framework");
path_to_mnist_cloth_dataset = getenv("path_to_mnist_cloth_dataset");

addpath(genpath(path_to_framework));
addpath(genpath(path_to_mnist_cloth_dataset));

%% Init parameters
verbose_level = 10;
normalice_data = true;
% vertical is throught each sample, allong all its features
% horizontal is throught the same features along all samples
% all is throught all matrix composition
% usually horizontal is used
normalice_mode = "horizontal2D";

% normalice refers to standardization or normalization

%% Load data

%{
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
0 => 10 T-shirt/top
%}

X = loadMNISTImages('dataset/t10k-images-idx3-ubyte');

if normalice_data
    [X, data_mean, data_std] = standardization(X, normalice_mode);
    % data_mean and data_std should be saved for new data incoming
end

Y = loadMNISTLabels('dataset/t10k-labels-idx1-ubyte');
Y(Y == 0) = 10;
Y = sparse_one_hot_encoding(Y, 10);

len_data = size(Y, 1);

%% Split sets
generate_rng(seed_rng);

[X_train, y_train, X_test, y_test, X_validation, y_validation] = split_data_2D(X, Y, 0.7, 0.1);

% test image read is correct even after split
index_image_test = 55;
image_test = reshape(X_test(:, index_image_test), 28, 28);
if ~normalice_data || normalice_mode ~= "all"
    imshow(image_test);  % if no standarization aplied should show the object
end
disp(y_test(index_image_test, :));



%% hyper parameters
generate_rng(seed_rng);

epochs = 100;
learning_rate = 0.002;
decay_rate_alpha = 0.1;
batch_size = 56;

% BatchNormalization() not combine with dropout, ...
% https://arxiv.org/pdf/1801.05134.pdf

sequential = Sequential({
    Dense(128, 'kaiming', 28 * 28), ...
    Activation("relu"), ...
    Dropout(0.3), ...
    Dense(64, 'kaiming'), ...
    Activation("relu"), ...
    Dropout(0.2), ...
    Dense(10), ...
    ActivationOnlyForward("softmax"), ...
});


nnConfig = NNConfig(epochs, learning_rate, batch_size, "softmax_cross_entropy");
nnConfig.decay_rate_alpha = decay_rate_alpha;

neural_network = NeuralNetwork(sequential, nnConfig);

%% train

history = neural_network.train(X_train, y_train, verbose_level);

fprintf("Mean error: %.5f\n", mean(history("error")));
plot(history("error"));

%% prediction
fprintf("\nPREDICTION\n");
raw_y_pred = neural_network.predict(X_test);
[~, idx_pred] = max(raw_y_pred);
[~, idx_real] = max(transpose(y_test));

accuracy = sum(idx_pred == idx_real)/length(idx_pred);
fprintf("Accuracy: %.4f\n", accuracy);
