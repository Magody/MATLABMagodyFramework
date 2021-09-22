function [reward, new_state, finish] = getRewardBestPath(state, action_selected, context) 

    terminal = context("terminal");
    reward_matrix = context("reward_matrix");
    reward = reward_matrix(state, action_selected);
    
    if reward == -1
        new_state = state;
    else
        new_state = action_selected;
    end
    
    finish = action_selected == terminal && reward >= 0;

end