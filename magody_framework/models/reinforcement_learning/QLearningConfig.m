%{
Made by: Danny Díaz
EPN - 2021
%}
classdef QLearningConfig < handle
    
    properties
        
        
        total_episodes;
        
        gamma = 1;
        initial_epsilon;
        
        gameReplayStrategy = 1;  % 1=reserved for each class, other number reserved is equitative for all
        experience_replay_reserved_space = 100;  % space per class
        
    end
    
    methods
        
        function self = QLearningConfig(gamma, epsilon, gameReplayStrategy, experience_replay_reserved_space, total_episodes)
            self.gamma = gamma;
            self.initial_epsilon = epsilon;
            self.gameReplayStrategy = gameReplayStrategy;
            self.experience_replay_reserved_space = experience_replay_reserved_space;
            self.total_episodes = total_episodes;
        end
    
    end
    
    
end

