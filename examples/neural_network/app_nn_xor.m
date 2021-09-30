%% Clean up
clear all;
clc;
close all;

seed_rng = 44;
%% Libs
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
addpath(genpath(path_to_framework));

%% Load data

X = [0, 0; 0, 1; 1, 0; 1, 1];
Y = [0; 1; 1; 0];

len_data = size(Y, 1);

%% Init parameters

generate_rng(seed_rng);
epochs = 100;
learning_rate = 0.01;
batch_size = 2;
verbose_level = 10;
decay_rate_alpha = 0.01;

% Dropout(0.5), ...

sequential_network = Sequential({
    Dense(5, "kaiming", 2), ...
    Activation("relu"), ...
    Dense(1), ...
    Activation("sigmoid") ...
});

nnConfig = NNConfig(epochs, learning_rate, batch_size, "mse");
nnConfig.decay_rate_alpha = decay_rate_alpha;

neural_network = NeuralNetwork(sequential_network, nnConfig);


%% train

history = neural_network.train(X, Y, verbose_level);

fprintf("Mean error: %.4f\n", mean(history("error")));
plot(history("error"));

%% prediction

fprintf("\nPREDICTION\n");
raw_y_pred = neural_network.predict(X);
y_pred = round(raw_y_pred);

accuracy = sum(y_pred(:) == Y)/len_data;
fprintf("Accuracy: %.4f\n", accuracy);

%% decision boundary plot

number_points = 100;

points = zeros([number_points*2, 3]);
index_points = 1;

for x=linspace(0, 1, number_points)
    for y=linspace(0, 1, number_points)
        
        z = [x; y];
        output = neural_network.sequential_network.forward(z);

        
        points(index_points, :) = [x, y, output(1,1)];
        index_points = index_points + 1;
    end
end

plot3(points(:, 1), points(:, 2), points(:, 3));