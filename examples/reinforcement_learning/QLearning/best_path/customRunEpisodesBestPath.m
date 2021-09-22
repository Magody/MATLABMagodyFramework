function history_episodes = customRunEpisodesBestPath(qLearning, functionGetReward, is_test, context, verbose_level)
    
    num_nodes = context("num_actions");
    Q_table = zeros([num_nodes,num_nodes]);
    
    history_episodes = containers.Map();
    
    history_q_val = zeros([1, qLearning.qLearningConfig.total_episodes]);
    history_scores = zeros([1, qLearning.qLearningConfig.total_episodes]);

    for episode=1:qLearning.qLearningConfig.total_episodes
        
        % rand world
        state = randi([1, num_nodes]);
        context("state") = state;
        
        context("Q_table") = Q_table;
    
        history_episode = qLearning.functionExecuteEpisode(qLearning, episode, is_test, functionGetReward, context, verbose_level-1);
        action = history_episode("action");
        update = history_episode("update");
        
        Q_table(state, action) = update;
        
            
        history_q_val(1, episode) = history_episode('max_q');
        history_scores(1, episode) = history_episode('score');

            
    end
    
    history_episodes('history_q_val') = history_q_val;
    history_episodes('Q_table') = Q_table;
    history_episodes('history_scores') = history_scores;
    
    
    
            
end

