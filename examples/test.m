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

shape = [300, 300, 5];
x = rand([shape, 10]);
convolutional = Convolutional([3, 3], 100, 0, 1, shape);

t1 = tic;
convolutional.forward(x);
t2 = toc(t1);
fprintf("Tiempo: %.4f [m]\n", t2/60);




