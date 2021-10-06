%{
Made by: Danny Díaz
EPN - 2021
%}
classdef QNeuralNetwork < QLearning
    
    properties
        
        % Neural network parameters
        shape_input;
        sequential_conv_network; % here is an optional network, {} for ignore it
        sequential_network;
        
        sequential_conv_network_target; 
        sequential_network_target;
        
        shape_output;
        
        nnConfig;
        
        % alpha decay
        alpha;
        
        % aux
        use_convolutional;
        actions_length;
    end
    methods
        
        function self = QNeuralNetwork(sequential_conv_network, sequential_network, nnConfig, qLearningConfig, functionExecuteEpisode)
            
            self = self@QLearning(qLearningConfig, functionExecuteEpisode);
            
            self.use_convolutional = ~isempty(sequential_conv_network.network);
    
            if self.use_convolutional
                self.shape_input = sequential_conv_network.network{1}.shape_input;
            else
                self.shape_input = sequential_network.network{1}.shape_input;
            end
            
            self.shape_output = sequential_network.network{end}.shape_output;
             
            self.actions_length = prod(self.shape_output);
            self.initGameReplay(self.actions_length);
            
            self.sequential_conv_network = sequential_conv_network;
            self.sequential_network = sequential_network;
            
            % theta freeze
            self.sequential_conv_network_target = sequential_conv_network; 
            self.sequential_network_target = sequential_network; 
            % NN and QNN config
            self.nnConfig = nnConfig;
            self.qLearningConfig = qLearningConfig;
            % decay epsilon/alpha
            self.epsilon = self.qLearningConfig.initial_epsilon;
            self.alpha = self.nnConfig.learning_rate;
            
            
        end
        
        function updateQNeuralNetworkTarget(self)
            % copy weights
            self.sequential_conv_network_target.network = self.sequential_conv_network.network;
            self.sequential_network_target.network = self.sequential_network.network; 
        end
        
        function history = train(self, X, Y, episode, verbose_level) %#ok<INUSD>
            context = containers.Map({'is_test'}, {false});
            % is trained as multiclass classification
            
            size_X = size(X);
            input_dim = length(size_X);
            
            history = containers.Map();
            
            history_errors = zeros([1, self.nnConfig.epochs]);
            
            len_data_train = size(X, input_dim);
            
            
            num_batchs = ceil(len_data_train/self.nnConfig.batch_size);

            for epoch=1:self.nnConfig.epochs
                error = 0;
                for index_data=1:self.nnConfig.batch_size:len_data_train
        
                    batch_range = index_data:min(len_data_train, index_data+self.nnConfig.batch_size-1);

                    if input_dim == 2
                        x = X(:, batch_range);
                    elseif input_dim == 3
                        x = X(:, :, batch_range);
                    elseif input_dim == 4
                        x = X(:, :, :, batch_range);
                    end
                    y = Y(batch_range, :);
                    
                    output = self.forwardFull(x, context);
                    
                    % error
                    error = error + sum(sum(self.nnConfig.functionLossCost(y', output), 1), 2)/self.nnConfig.batch_size;

                    if isnan(error) || sum(sum(isnan(output))) > 0
                        disp("Error: gradient exploding or vanishing");
                    end
                    
                    grad = self.nnConfig.functionLossGradient(transpose(y), output);
                    grad = self.sequential_network.backward(grad, self.alpha);
                    self.sequential_conv_network.backward(grad, self.alpha);

                end
                
                
                error = error / num_batchs;
                
                history_errors(1, epoch) = error;
                
            end
            
            % epsilon decay
            self.updateEpsilonDecay(1, episode);

            % alpha decay
            self.alpha = self.nnConfig.learning_rate/(1 + self.nnConfig.decay_rate_alpha * episode);
                

            history('history_errors') = history_errors;
        end
        
        function output = forwardFull(self, x, context)
            features = x;
            
            if self.use_convolutional
                features = self.sequential_conv_network.forward(x, context);
            end

            output = self.sequential_network.forward(features, context);
                     
        end
        
        
        function y_pred = predict(self, X)
            context = containers.Map({'is_test'}, {true});
            y_pred = self.forwardFull(X, context);
        end
        
        function [max_q, action_index] = selectAction(self, state, is_test)
            context = containers.Map({'is_test'}, {is_test});
            Qval = self.forwardFull(state, context)'; 
            [max_q, action_index] = QLearning.selectActionQEpsilonGreedy(Qval, self.epsilon, self.shape_output, is_test);
            
        end
        
        function history_learning = learnFromExperienceReplay(self, episode, verbose_level)
            
            context = containers.Map({'is_test'}, {false});
            
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
            
            if length(valid_replay) < self.nnConfig.batch_size
                return;
            end
            
            
            [~, idx] = sort(rand(length(valid_replay), 1));
            randIdx = idx(1:self.nnConfig.batch_size);
            
            input_dataX = self.shape_input;
            if input_dataX(1, 2) == 1
                input_dataX = self.shape_input(1);
            end
            
            dataX = zeros([input_dataX, self.nnConfig.batch_size]);
            dataY = zeros(self.nnConfig.batch_size, self.actions_length);
            
            input_dim = length(input_dataX);

            % Computations for the minibatch
            for numExample=1:self.nnConfig.batch_size

                % Getting the value of Q(s, a)
                s = valid_replay{randIdx(numExample)}.state;
                s_Qval = self.forwardFull(s, context);
                
                % Getting the value of max_a_Q(s',a')
                s_prime = valid_replay{randIdx(numExample)}.new_state;
                
                features = self.sequential_conv_network_target.forward(s_prime, context);
                s_prime_Qval = self.sequential_network_target.forward(features, context);
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
                    update_er = reward_er + self.qLearningConfig.gamma*maxQval_er;
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
            
            history = self.train(dataX, dataY, episode, verbose_level-1);
            
            
            history_errors = history('history_errors');
            history_learning('mean_cost') = mean(history_errors);
            
            history_learning('learned') = true;
            
        end
    
    end
    
    
end

