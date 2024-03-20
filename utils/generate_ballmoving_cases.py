import csv
import numpy as np
import datetime
import pandas as pd

# Generate random ballmoving scenarios
def generate_random_ball_moving(num_balls, num_rooms):
    
    # generate initial state: n rooms, n+1 state (first one is the location of the robot)
    initial_state = np.random.randint(low=1, high=num_rooms+1, size = num_balls + 1)

    # random goal state 
    goal_state = np.random.randint(low=1, high=num_rooms+1, size = num_balls)

    return initial_state, goal_state


if __name__=="__main__":

    # Generate cases
    num_balls = 3
    num_rooms = 4
    maximal_generation_attempts = 10000
    desired_num_cases = 200

    test_case_count = 0

    # initial state that is included in the prompt examples
    if num_balls == 3:

        initial_state_list = np.array([[2, 3, 2, 4],[1, 1, 3, 2]])
        goal_state_list = np.array([[1, 2, 3], [2, 1, 4]])
        scenario_list = np.hstack((initial_state_list, goal_state_list))

    elif num_balls == 4:
        initial_state_list = np.array([[3, 1, 3, 1, 2],[4, 2, 4, 1, 3]])
        goal_state_list = np.array([[3, 2, 4, 4], [3, 4, 2, 1]])
        scenario_list = np.hstack((initial_state_list, goal_state_list))

    for i in range(maximal_generation_attempts):

        # if enough samples
        if test_case_count >= desired_num_cases:
            break

        initial_state, goal_state = generate_random_ball_moving(num_balls, num_rooms)

        print(initial_state, goal_state)

        # check if goal state is the same as initial state
        if np.array_equal(goal_state, initial_state[1:]):

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
    file_name = "test_scenarios_" + str(num_balls) +"_balls_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
    file_name_npy_initial_state = "test_scenarios_" + str(num_balls) +"_balls_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_initial_state.npy"
    file_name_npy_goal_state = "test_scenarios_" + str(num_balls) +"_balls_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_goal_state.npy"
    
    df = pd.DataFrame({"initial_state": initial_state_list.tolist(), "goal_state" : goal_state_list.tolist()})
    df.to_csv(file_name, index=False)

    np.save(file_name_npy_initial_state, initial_state_list)
    np.save(file_name_npy_goal_state, goal_state_list)

