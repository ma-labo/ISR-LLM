import csv
import numpy as np
import datetime
import pandas as pd

# Generate random blockworld scenarios
def generate_random_blockworld(num_blocks):

    # max number of layer is the same as the block numbers
    max_layer = int(num_blocks)
    
    # randomize block positions
    remaining_blocks = int(num_blocks)
    num_block_per_layer = np.zeros(num_blocks, dtype = int)
    # random block order to assign
    block_order = np.random.permutation(num_blocks)
    
    # generate initial state
    initial_state = np.zeros(num_blocks,  dtype = int) +50

    # determin blocks from bottom layer to top layer
    for i in range(max_layer):

        # if no remaining blocks need to be assigned
        if remaining_blocks <= 0:

            break
        
        # if first layer: low - 1 block, high - num_blocks
        if i == 0:

            num_block_per_layer[i] = np.random.randint(low=1, high=num_blocks+1)
            remaining_blocks = int(remaining_blocks - num_block_per_layer[i])

            # assgin blocks that are on the table
            for j in range(num_block_per_layer[i]):
                initial_state[block_order[j]] = 0

        # if not: low - 1 block, high - min(num of blocks in previous layer, remaining_blocks)
        else:

            num_block_pre_layer = num_block_per_layer[i-1]
            used_blocks = num_blocks - remaining_blocks
            num_block_per_layer[i] = np.random.randint(low=1, high=min(num_block_pre_layer,remaining_blocks) +1)
            remaining_blocks = int(remaining_blocks - num_block_per_layer[i])

            # assign blocks on previous layer
            for j in range(num_block_per_layer[i]):
                block = block_order[j + used_blocks]
                initial_state[block] = block_order[j + used_blocks - num_block_pre_layer] + 1 # index 0 should be block 1

    # random goal state 
    goal_state = np.random.permutation(num_blocks) + 1 # index 0 should be replace by 1

    return initial_state, goal_state


# transform goal state to the same representation of block state for comparision
def transform_goal_state(goal_state):

    num_blocks = len(goal_state)
    goal_state_trans = np.zeros(num_blocks, dtype= int)

    # from top to bottom
    for i in range(num_blocks):

        if i < num_blocks - 1:

            block_current_layer = goal_state[i]
            block_next_layer = goal_state[i+1]
            goal_state_trans[block_current_layer-1] = block_next_layer

        # last block on table
        else:

            block_current_layer = goal_state[i]
            goal_state_trans[block_current_layer -1] = 0

    return goal_state_trans


if __name__=="__main__":

    # Generate cases
    num_blocks = 3
    maximal_generation_attempts = 10000
    desired_num_cases = 200

    test_case_count = 0

    # initial state that is included in the prompt examples
    if num_blocks == 3:
        initial_state_list = np.array([[0, 0, 1],[0, 3, 0]])
        goal_state_list = np.array([[1, 2, 3], [2, 1, 3]])
        scenario_list = np.hstack((initial_state_list, goal_state_list))

    elif num_blocks == 4:
        initial_state_list = np.array([[0, 4, 1, 0],[0, 4, 0, 0]])
        goal_state_list = np.array([[3, 2, 1, 4], [3, 4, 2, 1]])
        scenario_list = np.hstack((initial_state_list, goal_state_list))

    for i in range(maximal_generation_attempts):

        # if enough samples
        if test_case_count >= desired_num_cases:
            break

        initial_state, goal_state = generate_random_blockworld(num_blocks)

        # check if goal state is the same as initial state
        goal_state_trans = transform_goal_state(goal_state)
        if np.array_equal(goal_state_trans, initial_state):

            is_repeated = True

        else:

            # check if repeated 
            scenario = np.hstack((initial_state, goal_state))
            is_repeated = any(np.equal(scenario, scenario_list).all(1))

        if is_repeated == False:

            initial_state_list = np.vstack((initial_state_list, initial_state))
            goal_state_list = np.vstack((goal_state_list, goal_state))  
            scenario_list = np.hstack((initial_state_list, goal_state_list))

            test_case_count = test_case_count + 1


    # write to csv file and npy files
    file_name = "test_scenarios_" + str(num_blocks) +"_blocks_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
    file_name_npy_initial_state = "test_scenarios_" + str(num_blocks) +"_blocks_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_initial_state.npy"
    file_name_npy_goal_state = "test_scenarios_" + str(num_blocks) +"_blocks_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_goal_state.npy"
    
    df = pd.DataFrame({"initial_state": initial_state_list.tolist(), "goal_state" : goal_state_list.tolist()})
    df.to_csv(file_name, index=False)

    np.save(file_name_npy_initial_state, initial_state_list)
    np.save(file_name_npy_goal_state, goal_state_list)

