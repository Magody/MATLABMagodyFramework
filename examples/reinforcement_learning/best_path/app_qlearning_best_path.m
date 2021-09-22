%% Clean up
clc;
clear all; %#ok<CLALL>
close all;

%% Libs
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
addpath(genpath(path_to_framework));

%% load data
generate_rng(44); 
%#ok<*NBRAK>

environment_graph_literal = [ ...
    [0, 1]; ... 
    [1, 5]; ... 
    [5, 6]; ... 
    [5, 4]; ... 
    [1, 2]; ... 
    [2, 3]; ... 
    [2, 7] ...
];

num_nodes = max(environment_graph_literal(:)) + 1;  % plus zero
edges = size(environment_graph_literal, 1);
names = string(0:7);

goal = 6;  % 8 -> 7 + offset
reward_matrix = -1 * ones([num_nodes, num_nodes]);

adjacency_matrix = zeros([num_nodes, num_nodes]);
for i=1:edges
    from = environment_graph_literal(i, 1) + 1;
    to = environment_graph_literal(i, 2) + 1;
    adjacency_matrix(from, to) = 1;
    adjacency_matrix(to, from) = 1;
    
    if to == goal
        reward_matrix(from, to) = 10;
    else
        reward_matrix(from, to) = 0;
    end
    
    if from == goal
        reward_matrix(to, from) = 10;
    else
        reward_matrix(to, from) = 0;
    end
end

reward_matrix(goal, goal) = 10;

disp(reward_matrix);
% check graph is correct representation of environment
G = digraph(adjacency_matrix, names);
plot(G);
%{
environment_graph = containers.Map();
environment_graph("0") = [1];
%}
%% Init general parameters
verbose_level = 10;
context = containers.Map();
context("reward_matrix") = reward_matrix;
context("num_actions") = num_nodes;
context("terminal") = goal;

%% Init hyper parameters
gamma = 0.8;
epsilon = 1;
total_episodes = 1000;
% no experience replay
gameReplayStrategy = 0;
experience_replay_reserved_space = 0;

qLearningConfig = QLearningConfig(gamma, epsilon, gameReplayStrategy, experience_replay_reserved_space, total_episodes);
qLearning = QLearning(qLearningConfig, @executeEpisodeBestPath);
qLearning.setCustomRunEpisodes(@customRunEpisodesBestPath);


%% Train

t_begin = tic;

history_episodes = qLearning.runEpisodes(@getRewardBestPath, false, context, verbose_level-1);   
    
t_end = toc(t_begin);
fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);

Q_table = history_episodes('Q_table')

%% plot training result
subplot(1, 2, 1);
plot(history_episodes('history_scores'));
title("Scores");
subplot(1, 2, 2);
plot(history_episodes('history_q_val'));
title("Max q");

%% test

state = 1;
best_path = [state];

fprintf("Best path for begin in %d\n", state);

for step=1:5
    [~, action] = QLearning.selectActionQEpsilonGreedy(Q_table(state, :), qLearning.epsilon, 8, true);
    state = action;
    best_path = [best_path, state];
end
disp(best_path);

