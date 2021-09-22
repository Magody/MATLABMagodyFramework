function history_episode = executeEpisodeBestPath(qLearning, episode, is_test, functionGetReward, context, verbose_level) %#ok<INUSD>
    

    history_episode = containers.Map();
    
    state = context("state");
    Q_table = context("Q_table");
    num_actions = context("num_actions");
    
    q_values_actual = Q_table(state, :);
    
    [~, action] = QLearning.selectActionQEpsilonGreedy(q_values_actual, qLearning.epsilon, num_actions, is_test);
    
    % context is modified by reference
    [reward, new_state, finish] = functionGetReward(state, action, context);
    
    q_values_next = Q_table(new_state, :);
    max_q = max(q_values_next);
    
    if max_q > 0
        score = sum(sum(Q_table/max(max(Q_table))));  % normalization
    else
        score = 0;
    end


    % learn
    if finish
        update = reward;
    else
        update = reward + qLearning.qLearningConfig.gamma * max_q;
    end
    
    history_episode('reward') = reward;
    history_episode('update') = update;
    history_episode('action') = action;
    history_episode('max_q') = max_q;
    history_episode('score') = score;
    
    
    
    
    qLearning.updateEpsilonDecay(1, episode);
    
end