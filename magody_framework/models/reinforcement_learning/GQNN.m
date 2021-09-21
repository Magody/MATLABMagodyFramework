classdef GQNN < handle
    
    properties
        
        q_neural_network;
        q_neural_network_target;
        
        useCustomRunEpisodes = false;
        functionCustomRunEpisodes;
        functionExecuteEpisode;
        
        % Game replay
        experience_replay_reserved_space = 100;  % space per class
        gameReplay;
        gameReplayCounter = 0;
        gameReplayStrategy = 1;  % 1=reserved for each class, other number reserved is equitative for all
        
        % aux
        index_action = [];
        actions;
        actions_length;
        
    end
    
    methods
        function self = GQNN(q_neural_network, gameReplayStrategy, experience_replay_reserved_space, functionExecuteEpisode)
            
            self.q_neural_network = q_neural_network;
            self.q_neural_network_target = q_neural_network;
            
            self.gameReplayStrategy = gameReplayStrategy;
            self.experience_replay_reserved_space = experience_replay_reserved_space;
            
            
            self.actions = 1:self.q_neural_network.network{end}.shape_output;  
            self.actions_length = length(self.actions);
            self.index_action = zeros([1, self.actions_length]);
            
            self.gameReplay = cell([1, self.experience_replay_reserved_space * self.actions_length]);

            self.functionExecuteEpisode = functionExecuteEpisode;
        end
        
        function setCustomRunEpisodes(self, functionCustomRunEpisodes)
            self.useCustomRunEpisodes = true;
            self.functionCustomRunEpisodes = functionCustomRunEpisodes;
        end
        
        function history_episodes = runEpisodes(self, total_episodes, functionGetReward, is_test, context, verbose_level)
         
            if self.useCustomRunEpisodes
                % custom runEpisodesFunction
                history_episodes = self.functionCustomRunEpisodes(self, total_episodes, functionGetReward, is_test, context, verbose_level);
            else
                
                % pass
                history_episodes = self.runEpisodesDefault(total_episodes, functionGetReward, is_test, context, verbose_level);
            
            end
            
        end
        
        
        
        
        
        function history_episodes = runEpisodesDefault(self, total_episodes, functionGetReward, is_test, context, verbose_level)
            
            
            history_episodes = containers.Map();
            history_rewards = zeros([1, total_episodes]);
            
            history_update_costs = cell([1, total_episodes]);
            
            
           
            for episode=1:total_episodes

                if (mod(episode, 10) == 0 || episode == 1 || episode == total_episodes)  && verbose_level >= 1
                    fprintf("Episode %d of %d, is test: %d\n", episode, total_episodes, is_test);
                end                
                
                history_episode = self.functionExecuteEpisode(self, episode, total_episodes, is_test, functionGetReward, context, verbose_level-1);
                
                history_rewards(1, episode) = history_episode('reward_cummulated');
                
                update_costs = history_episode('update_costs');
                history_update_costs{1, episode} = update_costs(:);

                if ~is_test
                    % QNN target strategy, for "stable" learning
                    self.updateQNeuralNetworkTarget();
                end
            end
            history_episodes('rewards') = history_rewards;
            history_episodes('update_costs') = history_update_costs;
                       
        end
        
        function updateQNeuralNetworkTarget(self)
            % copy weights
            self.q_neural_network_target.conv_network = self.q_neural_network.conv_network;
            self.q_neural_network_target.network = self.q_neural_network.network; 
        end
        
        
        function saveExperienceReplay(self, state, action, reward, new_state, is_terminal)
            
            
            self.gameReplayCounter = self.gameReplayCounter + 1;
            if self.gameReplayStrategy == 1
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
            
        function history_learning = learnFromExperienceReplay(self, episode, total_episodes, verbose_level)
            
            history_learning = containers.Map();
            history_learning('learned') = false;
            
            valid_replay = getCellsNotEmpty(self.gameReplay);
            %{
            pending optimice with flag
            if self.gameReplayCounter < length(self.gameReplay)
                valid_replay = getCellsNotEmpty(self.gameReplay);
            else
                valid_replay = self.gameReplay;
            end
            %}
            
            if length(valid_replay) < self.q_neural_network.nnConfig.batch_size
                return;
            end
            
            
            [~, idx] = sort(rand(length(valid_replay), 1));
            randIdx = idx(1:self.q_neural_network.nnConfig.batch_size);
            
            dataX = zeros([self.q_neural_network.shape_input, self.q_neural_network.nnConfig.batch_size]);
            dataY = zeros(self.q_neural_network.nnConfig.batch_size, self.actions_length);
            
            input_dim = length(self.q_neural_network.shape_input);

            % Computations for the minibatch
            for numExample=1:self.q_neural_network.nnConfig.batch_size

                % Getting the value of Q(s, a)
                s = valid_replay{randIdx(numExample)}.state;
                
                
                s_Qval = self.q_neural_network.forwardFull(s);
                
                % Getting the value of max_a_Q(s',a')
                s_prime = valid_replay{randIdx(numExample)}.new_state;
                s_prime_Qval = self.q_neural_network_target.forwardFull(s_prime);
                maxQval_er = max(s_prime_Qval);
                
                % selected action and reward
                action_er = valid_replay{randIdx(numExample)}.action;
                reward_er = valid_replay{randIdx(numExample)}.reward;

                is_terminal = valid_replay{randIdx(numExample)}.is_terminal;
                if is_terminal
                    % Terminal state
                    update_er = reward_er;
                else
                    % Non-terminal state
                    update_er = reward_er + self.q_neural_network.qnnConfig.gamma*maxQval_er;
                end

                % Data for training
                
                if input_dim == 1
                    dataX(:, numExample) = s;
                elseif input_dim == 2
                    dataX(:, :, numExample) = s;
                elseif input_dim == 3
                    dataX(:, :, :, numExample) = s;
                end
                
                dataY(numExample, :) = s_Qval;
                dataY(numExample, action_er) = update_er;
                
            end
            
            
            history = self.q_neural_network.train(dataX, dataY, episode, total_episodes, verbose_level-1);
            
            history_learning('mean_cost') = mean(history('history_errors'));
            
            history_learning('learned') = true;
            
        end
        
    end
end

