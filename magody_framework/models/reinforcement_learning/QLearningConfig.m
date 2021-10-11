%{
Made by: Danny Díaz
EPN - 2021
%}
classdef QLearningConfig < handle
    
    properties
        
        
        total_episodes;
        total_episodes_test = 1;
        gamma = 1;
        initial_epsilon;
        
        gameReplayStrategy = 1;  % 1=reserved for each class, other number reserved is equitative for all
        experience_replay_reserved_space = 100;  % space per class
        
        interval_for_learning = 10;
        rewards = struct('correct', 1, 'incorrect', -1);
        
    end
    
    methods
        
        function self = QLearningConfig(gamma, epsilon, gameReplayStrategy, ...
                experience_replay_reserved_space, total_episodes, interval_for_learning, ...
                rewards)
            self.gamma = gamma;
            self.initial_epsilon = epsilon;
            self.gameReplayStrategy = gameReplayStrategy;
            self.experience_replay_reserved_space = experience_replay_reserved_space;
            self.total_episodes = total_episodes;
            self.interval_for_learning = interval_for_learning;
            self.rewards = rewards;
        end
    
    end
    
    
end

