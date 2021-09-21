%% Clean up
clear all;
clc;
close all;

%% Libs
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
path_to_mnist_cloth_dataset = "/home/magody/programming/MATLAB/deep_learning_from_scratch/examples/dataset";
dir_base_models_result = "/home/magody/programming/MATLAB/deep_learning_from_scratch/examples/convolutional_neural_network/models_result";

addpath(genpath(path_to_framework));
addpath(genpath(path_to_mnist_cloth_dataset));

seed_rng = 44;

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
generate_rng(seed_rng);

[X_train, y_train, X_test, y_test, X_validation, y_validation] = split_data_2D(X, Y, 0.01, 0.1);

X_train = reshape(X_train, [28, 28, 1, size(y_train, 1)]);
X_test = reshape(X_test, [28, 28, 1, size(y_test, 1)]);
X_validation = reshape(X_validation, [28, 28, 1, size(y_validation, 1)]);
% test image read is correct even after split
index_image_test = 55;
image_test = reshape(X_test(:, :, 1, index_image_test), 28, 28);
imshow(image_test);
disp(y_test(index_image_test, :));

%% Init parameters
class_names = ["Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot", "T-shirt"];

shape_input = [28, 28, 1];
verbose_level = 10;
model_name = "debug";
dir_model = dir_base_models_result + "/" + model_name;
dir_vars = dir_model + "/vars/";
dir_figures = dir_model + "/figures/";
% creates the directory if it doesnt exists
[status, msg, msgID] = mkdir(dir_model); %#ok<ASGLU>
[status, msg, msgID] = mkdir(dir_vars); %#ok<ASGLU>
[status, msg, msgID] = mkdir(dir_figures);

%% hyper parameters
generate_rng(seed_rng);
epochs = 10;
learning_rate = 0.01;
decay_rate_alpha = 0.1;
batch_size = 32;

conv_sequential = Sequential({...
    Convolutional([3, 3], 32, 0, 1, shape_input), ...
    Activation("relu"), ....
    Pooling("max"), ...
    Reshape(), ...
});

input_dense = prod(conv_sequential.shape_output);
   
sequential = Sequential({
    Dense(100, input_dense), ...
    Activation("relu"), ...
    Dense(10), ...
    ActivationOnlyForward("softmax"), ...
});


nnConfig = NNConfig(epochs, learning_rate, batch_size, "softmax_cross_entropy");
nnConfig.decay_rate_alpha = decay_rate_alpha;

convolutional_neural_network = ConvolutionalNeuralNetwork(conv_sequential, sequential, nnConfig);

%% train

t_begin = tic;

history = convolutional_neural_network.train(X_train, y_train, verbose_level);

t_end = toc(t_begin);
fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);

save(dir_vars + "cnn", "convolutional_neural_network");
save(dir_vars + "history", "history");
save(dir_vars + "dir_vars", "dir_vars");
save(dir_vars + "verbose_level", "verbose_level");




%% plot train results

figure(1);
plot(history("error"));

saveas(gcf, dir_figures + "loss.png");


figure(2);
times_conv_network_forward = history('times_conv_network_forward');
times_network_forward = history('times_network_forward');
times_conv_network_backward = history('times_conv_network_backward');
times_network_backward = history('times_network_backward');

subplot(2,2,1);
plot(sum(times_conv_network_forward, 2));
subplot(2,2,2);
plot(sum(times_network_forward, 2));
subplot(2,2,3);
plot(sum(times_conv_network_backward, 2));
subplot(2,2,4);
plot(sum(times_network_backward, 2));
saveas(gcf, dir_figures + "times.png");

%% prediction
fprintf("\nPREDICTION\n");
raw_y_pred = convolutional_neural_network.predict(X_test);
[~, idx_pred] = max(raw_y_pred);
[~, idx_real] = max(transpose(y_test));

accuracy = sum(idx_pred == idx_real)/length(idx_pred);
fprintf("Accuracy: %.4f\n", accuracy);

%% TSNE, dimension reduction. Table of features and plots

fprintf("Generating table of features\n");
len_data_train = size(X_train, 4);
features_matrix = zeros([len_data_train, convolutional_neural_network.sequential_conv_network.shape_output]);
classes = {};




for index_data=1:len_data_train
    
    x = X_train(:, :, :, index_data);
    [~, y] = max(y_train(index_data, :));  % argmax
    features = convolutional_neural_network.sequential_conv_network.forward(x);
    
    
    classes{index_data, 1} = char(class_names(y));
    features_matrix(index_data, :) = features';
            
end

%% TSNE plots
% features_matrix = table2array(features_table);

options = containers.Map();
options('limit_samples') = 10000;
options('plot_point_size') = 6;
options('include3D') = false;
options('save') = true;

options('dir') = dir_figures;
options('algorithms') =  [ ...
    struct('distance', 'euclidean','plot', struct('title', 'Euclidean')), ...
    struct('distance', 'chebychev','plot', struct('title', 'Chebychev')), ...
];

history_tsne = generateTSNE(features_matrix, classes, options, verbose_level-1);
k = keys(history_tsne);
v = values(history_tsne);
for i=1:length(k)
    fprintf("Results for algorithm %s: ", k{i});
    disp(v{i});
end
