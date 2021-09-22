classdef QLearning < handle
    
    properties
        
        useCustomRunEpisodes = false;
        functionCustomRunEpisodes;
        functionExecuteEpisode;
        
        % Config
        qLearningConfig;
        
        % Game replay
        gameReplay;
        gameReplayCounter = 0;
        
        % aux
        index_action = [];
        
        % epsilon decay
        epsilon;
        
    end
    
    methods 
       
        function self = QLearning(qLearningConfig, functionExecuteEpisode)
            self.qLearningConfig = qLearningConfig;
            self.epsilon = self.qLearningConfig.initial_epsilon;
            self.functionExecuteEpisode = functionExecuteEpisode;
            
        end
        
        function initGameReplay(self, actions_length)
            self.index_action = zeros([1, actions_length]);
            self.gameReplay = cell([1, self.qLearningConfig.experience_replay_reserved_space * actions_length]);
 
        end
        
        function setCustomRunEpisodes(self, functionCustomRunEpisodes)
            self.useCustomRunEpisodes = true;
            self.functionCustomRunEpisodes = functionCustomRunEpisodes;
        end
        
        function history_episodes = runEpisodes(self, functionGetReward, is_test, context, verbose_level)
         
            if self.useCustomRunEpisodes
                % custom runEpisodesFunction
                history_episodes = self.functionCustomRunEpisodes(self, functionGetReward, is_test, context, verbose_level);
            else
                % pass
                history_episodes = self.runEpisodesDefault(functionGetReward, is_test, context, verbose_level);
            end
            
        end
        
        function history_episodes = runEpisodesDefault(self, functionGetReward, is_test, context, verbose_level)
            
            
            history_episodes = containers.Map();
            history_rewards = zeros([1, self.qLearningConfig.total_episodes]);
            
            history_update_costs = cell([1, self.qLearningConfig.total_episodes]);
            
           
            for episode=1:total_episodes

                if (mod(episode, 10) == 0 || episode == 1 || episode == self.qLearningConfig.total_episodes)  && verbose_level >= 1
                    fprintf("Episode %d of %d, is test: %d\n", episode, self.qLearningConfig.total_episodes, is_test);
                end                
                
                history_episode = self.functionExecuteEpisode(self, episode, is_test, functionGetReward, context, verbose_level-1);
                
                history_rewards(1, episode) = history_episode('reward_cummulated');
                
                update_costs = history_episode('update_costs');
                history_update_costs{1, episode} = update_costs(:);

                if ~is_test
                    % QNN target strategy, for "stable" learning
                    % self.updateQNeuralNetworkTarget();
                end
            end
            history_episodes('rewards') = history_rewards;
            history_episodes('update_costs') = history_update_costs;
                       
        end
        
        
        
        function saveExperienceReplay(self, state, action, reward, new_state, is_terminal)
            
            self.gameReplayCounter = self.gameReplayCounter + 1;
            if self.qLearningConfig.gameReplayStrategy == 1
                self.index_action(action) = self.index_action(action) + 1;
                index_experience_replay = mod(self.index_action(action), self.experience_replay_reserved_space);
                if index_experience_replay == 0
                    index_experience_replay = self.experience_replay_reserved_space;
                end

                offset = (action-1) * self.experience_replay_reserved_space;
                index_replay = offset+index_experience_replay;
                
            else
                index_replay = mod(self.gameReplayCounter, size(self.gameReplay, 1));
                if index_replay == 0
                   index_replay =  size(self.gameReplay, 1);
                end
                
            end
            
            self.gameReplay{1, index_replay} = struct('state', state, 'action', action, 'reward', reward, 'new_state', new_state, 'is_terminal', is_terminal);   %[state(:)', action, reward, new_state(:)'];
        end
        
        function updateEpsilonDecay(self, mode, episode)
            % mode=1 -> interpolate between initial_epsilon and 0.01
            if mode == 1
                self.epsilon =  max(0.01, self.qLearningConfig.initial_epsilon * log(exp(1) - ((exp(1) - 1) * (episode/self.qLearningConfig.total_episodes))));
            end
        end
        
        
    end
    
    methods (Static)
        
        
        
        function [max_q, action_index] = selectActionQEpsilonGreedy(Qval, epsilon, num_actions, is_test)
            [max_q, idx] = max(Qval); % obtengo indice de Qmax a partir de vector Qval

            if ~is_test
                % Epsilon-greedy action selection
                % Inicialmente hace solo exploracion, luego el valor de epsilon se va reduciendo a medida que tengo mas informacion
                % Si rand <= epsilon, obtengo un Q de manera aleatoria, el cual será diferente a Qmax (exploracion)

                v = rand;
                if v <= epsilon       
                    full_action_list = 1:num_actions;
                    
                    actionList = full_action_list(full_action_list ~= idx);      %  Crea lista con las acciones q no tienen Qmax
                    idx_valid_action = randi([1 length(actionList)]);
                    idx = full_action_list(actionList(idx_valid_action));
                end
            end
            action_index = idx; 
        end
    end
    
end