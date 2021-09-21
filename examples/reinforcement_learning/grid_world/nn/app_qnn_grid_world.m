%% Clean up
clc;
clear all; %#ok<CLALL>
close all;

%% Libs
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
addpath(genpath(path_to_framework));

%% Init general parameters
generate_rng(44);
verbose_level = 10;

matrix_mask = [
    [1, 1, 1]; 
    [1, -1, 1]; 
    [0, -3, 1]
];
matrix = matrix_mask;
terminals = [[2, 2]; [3, 3]];
position = [3, 1];
matrix(position(1), position(2)) = 2;

input_size = (size(matrix, 1) * size(matrix, 2));
initial_state = reshape(matrix(:), [1 input_size]);
m = length(matrix);
n = length(matrix(1, :));

rewards = containers.Map({'collision', 'positive', 'negative', 'visited'}, ...
    {-4, 3, -1, -3});

context = containers.Map( ...
        {'matrix_mask', 'initial_matrix', 'initial_position', ...
            'initial_state', 'm', 'n', ...
            'terminals', 'input_size', ...
            'rewards'}, ...
        {matrix_mask, matrix, position, ...
        initial_state, m, n, ...
        terminals, input_size, rewards});
    
%% Init Q neural network and its hyper parameters

episodes_train = 100;
episodes_test = 1;
epochs = 2; % epochs inside each NN
learning_rate = 0.001;
batch_size = 32;
gamma = 0.99;
epsilon = 0.5;
experience_replay_reserved_space = 50;
gameReplayStrategy = 1;
loss_type = "mse";

conv_network = {};  % this app only will use feed forward network
sequential = Sequential({
    Dense(16, input_size), ...
    Activation("relu"), ...
    Dense(16), ...
    Activation("relu"), ...
    Dense(4), ...
});
network = sequential.network;

nnConfig = NNConfig(epochs, learning_rate, batch_size, loss_type);
nnConfig.decay_rate_alpha = 0.01;

qnnConfig = QNNConfig(gamma, epsilon);
    
q_neural_network = QNeuralNetwork(conv_network, network, nnConfig, qnnConfig);    

gqnn = GQNN(q_neural_network, sequential.shape_input, gameReplayStrategy, ...
            experience_replay_reserved_space, @executeEpisodeGridWorld);

%% Train
t_begin = tic;
history_episodes_train = gqnn.runEpisodes(episodes_train, @getRewardGridWorld, false, context, verbose_level-1);
t_end = toc(t_begin);
fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);

figure(1);
subplot(1,2,1)
plot(history_episodes_train('rewards'));

linear_update_costs = [];
update_costs_by_episode = history_episodes_train('update_costs');

for i=1:length(update_costs_by_episode)
    costs = update_costs_by_episode{i};
    linear_update_costs = [linear_update_costs; costs(:)]; 
end
subplot(1,2,2)
plot(linear_update_costs);

%% Test

history_episodes_test = gqnn.runEpisodes(episodes_test, @getRewardGridWorld, true, context, verbose_level-1);
mean_reward = mean(history_episodes_test('rewards'));
fprintf("Mean reward: %.4f\n", mean_reward);
