import csv
import numpy as np
import datetime
import pandas as pd

# Generate random cooking scenarios
def generate_random_cooking(num_pot, num_ingredient):

    goal_state = np.zeros((num_pot, num_ingredient),dtype =int)

    # random goal state 
    for i in range(num_pot):

        # random number of ingredients (2-4)
        n_ingre = np.random.randint(low=2, high=5)
        # random choice of ingredients
        ingre_index = np.random.choice(6, size = n_ingre, replace = False)
        goal_state[i, ingre_index] = 1

    return goal_state


if __name__=="__main__":

    # Generate cases
    num_pot = 4
    num_ingredient = 6
    maximal_generation_attempts = 10000
    desired_num_cases = 200

    test_case_count = 0

    # initial state that is included in the prompt examples
    if num_pot == 3:

        goal_state_list = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0]])
        scenario_list = goal_state_list.reshape(1, num_ingredient*num_pot)

    elif num_pot == 4:

        goal_state_list = np.array([[0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1]])
        scenario_list = goal_state_list.reshape(1, num_ingredient*num_pot)

    for i in range(maximal_generation_attempts):

        # if enough samples
        if test_case_count >= desired_num_cases:
            break

        goal_state = generate_random_cooking(num_pot, num_ingredient)

        # check if repeated 
        scenario = goal_state.reshape(1, num_ingredient*num_pot)
        is_repeated = any(np.equal(scenario, scenario_list).all(1))

        if is_repeated == False:

            scenario_list = np.vstack((scenario_list, scenario))
            test_case_count = test_case_count + 1


    # write to csv file and npy files
    file_name = "test_scenarios_" + str(num_pot) +"_pots_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
    file_name_npy_goal_state = "test_scenarios_" + str(num_pot) +"_pots_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_goal_state.npy"
    
    df = pd.DataFrame({"goal_state" : scenario_list.tolist()})
    df.to_csv(file_name, index=False)

    np.save(file_name_npy_goal_state, scenario_list)

