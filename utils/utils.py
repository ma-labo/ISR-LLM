import os
import numpy as np
import re

# load test scenarios according to domian type
def load_test_scenarios(args):

    test_file_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../test_scenario/") 

    print("Loading test scenarios for " + args.domain +" domian with " + str(args.num_objects) + " objects...")

    if args.domain == 'blocksworld':

        if args.num_objects == 3:

            initial_state_file = test_file_root + "blocksworld/test_scenarios_3_blocks_initial_state.npy"
            goal_state_file = test_file_root + "blocksworld/test_scenarios_3_blocks_goal_state.npy"
            initial_state = np.load(initial_state_file)
            goal_state = np.load(goal_state_file)

        elif args.num_objects == 4:

            initial_state_file = test_file_root + "blocksworld/test_scenarios_4_blocks_initial_state.npy"
            goal_state_file = test_file_root + "blocksworld/test_scenarios_4_blocks_goal_state.npy"
            initial_state = np.load(initial_state_file)
            goal_state = np.load(goal_state_file)

    elif args.domain == 'ballmoving':

        if args.num_objects == 3:

            initial_state_file = test_file_root + "ballmoving/test_scenarios_3_balls_initial_state.npy"
            goal_state_file = test_file_root + "ballmoving/test_scenarios_3_balls_goal_state.npy"
            initial_state = np.load(initial_state_file)
            goal_state = np.load(goal_state_file)

        elif args.num_objects == 4:

            initial_state_file = test_file_root + "ballmoving/test_scenarios_4_balls_initial_state.npy"
            goal_state_file = test_file_root + "ballmoving/test_scenarios_4_balls_goal_state.npy"
            initial_state = np.load(initial_state_file)
            goal_state = np.load(goal_state_file)

    elif args.domain == 'cooking':

        if args.num_objects == 3:

            goal_state_file = test_file_root + "cooking/test_scenarios_3_pots_goal_state.npy"
            goal_state = np.load(goal_state_file)
            initial_state = np.zeros(goal_state.shape, dtype=int)

        elif args.num_objects == 4:

            goal_state_file = test_file_root + "cooking/test_scenarios_4_pots_goal_state.npy"
            goal_state = np.load(goal_state_file)
            initial_state = np.zeros(goal_state.shape, dtype=int)

    else:

        raise ValueError("Unknown domain.")


    return initial_state, goal_state

# extract init and goal state from generated PDDL file
def extract_state_pddl(pddl_problem, domain):

    if domain == 'blocksworld':

        pddl_init_state = pddl_problem.split('(:init', 1)
        pddl_init_state = pddl_init_state[1]
        pddl_init_state = pddl_init_state.split('empty)\n', 1)
        pddl_init_state = pddl_init_state[1]
        pddl_init_state = pddl_init_state.split('\n(clear', )
        pddl_init_state = pddl_init_state[0]

        pddl_goal_state = pddl_problem.split('(:goal', 1)
        pddl_goal_state = pddl_goal_state[1]
        pddl_goal_state = pddl_goal_state.split('(and', 1)
        pddl_goal_state = pddl_goal_state[1] 
        pddl_goal_state = pddl_goal_state.split('))\n', 1)
        pddl_goal_state = pddl_goal_state[0]
        pddl_goal_state = pddl_goal_state + ')' 

    elif domain == 'ballmoving':

        pddl_init_state = pddl_problem.split('(:init', 1)
        pddl_init_state = pddl_init_state[1]
        pddl_init_state = pddl_init_state.split('empty)\n', 1)
        pddl_init_state = pddl_init_state[1]
        pddl_init_state = pddl_init_state.split('\n)', )
        pddl_init_state = pddl_init_state[0]

        pddl_goal_state = pddl_problem.split('(:goal', 1)
        pddl_goal_state = pddl_goal_state[1]
        pddl_goal_state = pddl_goal_state.split('(and', 1)
        pddl_goal_state = pddl_goal_state[1] 
        pddl_goal_state = pddl_goal_state.split('))\n', 1)
        pddl_goal_state = pddl_goal_state[0]
        pddl_goal_state = pddl_goal_state + ')' 

    elif domain == 'cooking':

        pddl_init_state = pddl_problem.split('(:init', 1)
        pddl_init_state = pddl_init_state[1]
        pddl_init_state = pddl_init_state.split('\n)', )
        pddl_init_state = pddl_init_state[0]

        pddl_goal_state = pddl_problem.split('(:goal', 1)
        pddl_goal_state = pddl_goal_state[1]
        pddl_goal_state = pddl_goal_state.split('(and', 1)
        pddl_goal_state = pddl_goal_state[1] 
        pddl_goal_state = pddl_goal_state.split('))\n', 1)
        pddl_goal_state = pddl_goal_state[0]
        pddl_goal_state = pddl_goal_state + ')' 

    return pddl_init_state, pddl_goal_state


# extract action description from response
def extract_action_description(action_sequence, domain):

    if domain == 'blocksworld':

        actions = re.findall(r'\(.*?\)', action_sequence)
        num_actions = len(actions)
        action_description = ""
        for i in range(num_actions):
            action_description = action_description + actions[i] + '\n'

    elif domain == 'ballmoving':

        actions = re.findall(r'\(.*?\)', action_sequence)
        filtered_actions = []
        for i in range(len(actions)):
            current_action = actions[i].split(' ')
            action_type = current_action[0][1:]
            if action_type == 'pick' or action_type == 'move' or action_type == 'drop':
                filtered_actions.append(actions[i])
        num_actions = len(filtered_actions)
        action_description = ""
        for i in range(num_actions):
            action_description = action_description + filtered_actions[i] + '\n'

    elif domain == 'cooking':

        actions = re.findall(r'\(.*?\)', action_sequence)
        num_actions = len(actions)
        action_description = ""
        for i in range(num_actions):
            action_description = action_description + actions[i] + '\n'

    return action_description

