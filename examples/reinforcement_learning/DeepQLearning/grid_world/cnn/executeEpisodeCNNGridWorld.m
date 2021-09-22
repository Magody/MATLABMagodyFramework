function history_episode = executeEpisodeCNNGridWorld(q_neural_network, episode, is_test, functionGetReward, context, verbose_level) %#ok<INUSD>
                         
    history_episode = containers.Map();
    
    update_counter = 0;
    update_costs = [];
    
    reward_cummulated = 0;
    step = 0;

    run_episode = true;
    state = context('initial_state');
    context('matrix') = context('initial_matrix');
    context('position') = context('initial_position');

    while run_episode

        [~, action] = q_neural_network.selectAction(state, is_test);

        
        % context is modified by reference
        [reward, new_state, finish] = functionGetReward(state, action, context);


        if ~is_test
            q_neural_network.saveExperienceReplay(state, action, reward, new_state, finish);  
            
            history_learning = q_neural_network.learnFromExperienceReplay(episode, verbose_level);
            if history_learning('learned')
                update_counter = update_counter + 1;
                update_costs(1, update_counter) = history_learning('mean_cost'); %#ok<AGROW>
            end

        end

        step = step + 1;
        
        %{
        
        %}
        
        reward_cummulated = reward_cummulated + reward;
        state = new_state;
        if finish
            run_episode = false;
        end
    end
    if ~is_test
        
        % QNN target strategy, for "stable" learning
        q_neural_network.updateQNeuralNetworkTarget();
    end
    
    history_episode('reward_cummulated') = reward_cummulated;
    history_episode('update_costs') = update_costs;
    
    
end