%% Clean up
clear all;
clc;
close all;

%% Libs
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
path_to_mnist_cloth_dataset = "/home/magody/programming/MATLAB/deep_learning_from_scratch/examples/dataset";

addpath(genpath(path_to_framework));
addpath(genpath(path_to_mnist_cloth_dataset));


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
% already normaliced [0, 1] gray scale
X = loadMNISTImages('dataset/t10k-images-idx3-ubyte');
Y = loadMNISTLabels('dataset/t10k-labels-idx1-ubyte');
Y(Y == 0) = 10;
Y = sparse_one_hot_encoding(Y, 10);

len_data = size(Y, 1);

%% Split sets

[X_train, y_train, X_test, y_test, X_validation, y_validation] = split_data_2D(X, Y, 0.7, 0.1);

% test image read is correct even after split
index_image_test = 55;
image_test = reshape(X_test(:, index_image_test), 28, 28);
imshow(image_test);
disp(y_test(index_image_test, :));

%% Init parameters
generate_rng(44);
verbose_level = 10;

%% hyper parameters

epochs = 100;
learning_rate = 0.001;
decay_rate_alpha = 0.01;
batch_size = 64;

sequential = Sequential({
    Dense(40, 28 * 28), ...
    Activation("relu"), ...
    Dense(40), ...
    Activation("sigmoid"), ...
    Dense(10), ...
    ActivationOnlyForward("softmax"), ...
});


nnConfig = NNConfig(epochs, learning_rate, batch_size, "softmax_cross_entropy");
nnConfig.decay_rate_alpha = decay_rate_alpha;

neural_network = NeuralNetwork(sequential, nnConfig);

%% train

history = neural_network.train(X_train', y_train, verbose_level);

plot(history("error"));

%% prediction
fprintf("\nPREDICTION\n");
raw_y_pred = neural_network.predict(X_test');
[~, idx_pred] = max(raw_y_pred);
[~, idx_real] = max(transpose(y_test));

accuracy = sum(idx_pred == idx_real)/length(idx_pred);
fprintf("Accuracy: %.4f\n", accuracy);
