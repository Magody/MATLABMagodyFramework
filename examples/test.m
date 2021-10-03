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

shape = [6, 6, 2];
x = rand([shape, 100]);
convolutional = Convolutional([3, 3], 2, 0, 1, shape);
convolutional.bias = rand(convolutional.shape_output);

%% Check forward

t1 = tic;
forward_vect0 = convolutional.forward(x);
t2 = toc(t1);
fprintf("Tiempo forward_vect0: %.4f [s]\n", t2);


t1 = tic;
forward_vect = convolutional.forwardSJLiXu(x);
t2 = toc(t1);
fprintf("Tiempo forward_vect: %.4f [s]\n", t2);

% CHECK THE CALC IS CORRECT
t1 = tic;
forward_novect = convolutional.forwardNoVectorized(x);
t2 = toc(t1);
fprintf("Tiempo forward_novect: %.4f [s]\n", t2);


checksum = sum(sum(sum(sum(abs(round(forward_vect)))))) == sum(sum(sum(sum(abs(round(forward_novect))))));

fprintf("0=%.4f\n" , sum(sum(sum(sum(forward_vect0)))))
fprintf("a=%.4f, b=%.4f\n", sum(sum(sum(sum(forward_vect)))), sum(sum(sum(sum(forward_novect)))));
if checksum
   disp("The forward is probably correct");
else
   disp("The forward is bad calculated"); 
end

%% check backward


original_kernels = convolutional.kernels;
original_bias = convolutional.bias;


convolutional.kernels = original_kernels;
convolutional.bias = original_bias;
forward_vect0 = convolutional.forward(x);
t1 = tic;
backward_vect = convolutional.backward(forward_vect0, 0.001);
t2 = toc(t1);
fprintf("Tiempo backward_vect: %.4f [s]\n", t2);


convolutional.kernels = original_kernels;
convolutional.bias = original_bias;
forward_novect = convolutional.forwardNoVectorized(x);
t1 = tic;
backward_novect = convolutional.backwardNoVectorized(forward_novect, 0.001);
t2 = toc(t1);
fprintf("Tiempo backward_novect: %.4f [s]\n", t2);


fprintf("a=%.4f, b=%.4f\n", sum(sum(sum(sum(abs(backward_vect))))), sum(sum(sum(sum(abs(backward_novect))))));


checksum = sum(sum(sum(sum(abs(round(backward_vect)))))) == sum(sum(sum(sum(abs(round(backward_novect))))));

if checksum
   disp("The backward is correct");
else
   disp("The backward is bad calculated"); 
end
