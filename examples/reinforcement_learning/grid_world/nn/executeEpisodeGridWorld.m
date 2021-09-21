function history_episode = executeEpisodeGridWorld(gqnn, episode, total_episodes, is_test, functionGetReward, context, verbose_level) %#ok<INUSD>
                         
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

        [Qval, action] = gqnn.q_neural_network.selectAction(state', is_test);

        % context is modified by reference
        [reward, new_state, finish] = functionGetReward(state, action, context);

        % dump custom debug
        if is_test || episode >= total_episodes * 0.99
            reshape(state, [3, 3])
            reshape(new_state, [3, 3])
            disp(Qval);
            disp("");
        end 


        if ~is_test
            gqnn.saveExperienceReplay(state, action, reward, new_state, finish);  
            
            history_learning = gqnn.learnFromExperienceReplay(episode, total_episodes, verbose_level);
            if history_learning('learned')
                update_counter = update_counter + 1;
                update_costs(1, update_counter) = history_learning('mean_cost'); %#ok<AGROW>
            end
        end

        step = step + 1;

        if mod(step, 40) == 0
           gqnn.updateQNeuralNetworkTarget(); 
        end

        reward_cummulated = reward_cummulated + reward;
        state = new_state;
        if finish
            run_episode = false;
        end
    end
    history_episode('reward_cummulated') = reward_cummulated;
    history_episode('update_costs') = update_costs;
    
    
end