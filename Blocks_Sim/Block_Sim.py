import os
import numpy as np
import re

class BlockSim(object):
    """
    Simple block simulator
    """
    def __init__(self):
        
        self.block_state = None
        self.clear_state = None
        self.goal_state = None
        self.constraint = None
        self.is_hand_empty = None

    # Generate a given specific block scenario
    def generate_scene_description(self, initial_state, goal_state, constraint = None):

        num_blocks = len(initial_state)

        print("Generating a given "+str(num_blocks)+" blocks scenario...")
        print("Initial state: ", initial_state)
        print("Goal state: ", goal_state)
        print("Constraint: ", constraint)

        # Description for LLM
        description = self.compose_description(num_blocks, initial_state, goal_state, constraint)

        return description

    # Compose task description 
    def compose_description(self, num_blocks, initial_state, goal_state, constraint):

        
        # Initial openning
        text_open = "I have " + str(num_blocks) + " blocks."
        
        # Initial state
        text_initial_state = " Initially: "
        for i in range(num_blocks):

            # if not on the table
            if initial_state[i] > 0:

                text_single = "Block b" + str(i+1) + " is on top of b" + str(initial_state[i])+". "

            # if on the table
            else:

                text_single = "Block b" + str(i+1) + " is on the table. "

            text_initial_state = text_initial_state + text_single

        # Condition
        text_condition = ""

        # Goal state
        text_goal = "Your goal is to move the blocks such that they are stacked in the order: "
        for i in range(num_blocks):

            if i < num_blocks - 1:

                text_single = "b"+ str(goal_state[i]) + " on b"+ str(goal_state[i+1]) + ", "

            else:

                text_single = "and b" + str(goal_state[i]) + " on table. "


            text_goal = text_goal + text_single

        # Constraint
        text_constraint = ""
        if constraint != None:
            text_constraint = "However: "
            for i in range(len(constraint)):

                constraint_single = constraint[i]
                text_single = "b" + str(constraint_single[0]) + " should not be on top of b" + str(constraint_single[1]) + " in all time. "
                text_constraint = text_constraint + text_single

        description = text_open + text_initial_state + text_condition + text_goal + text_constraint

        return description

    # Initialize simulation state
    def initialize_state(self, initial_state, goal_state, constraint):

        self.block_state = initial_state.copy() # use copy for avoiding reference
        self.goal_state = goal_state
        self.constraint = constraint
        self.num_blocks = len(initial_state)
        self.is_hand_empty = True

        # get clear state: 0 is unclear, 1 is clear, -1 means the block is in hand
        self.clear_state = self.determine_clear_state(block_state = self.block_state)

    # Determine clear state based on given state
    def determine_clear_state(self, block_state):

        num_blocks = len(block_state)
        clear_state = np.ones(num_blocks, dtype= int)

        for i in range(num_blocks):

            block_under = block_state[i]

            # if block is in hand (-1)
            if block_under == -1:

                clear_state[i] = -1

            # if is on another block, then the block under is not clear
            elif block_under != 0: 

                clear_state[block_under-1] = 0

        return clear_state


    # Simulate given action sequence
    def simulate_actions(self, action_sequence, test_log_file_path):

        actions = re.findall(r'\(.*?\)', action_sequence)
        print(actions)
        num_actions = len(actions)

        is_error = None
        is_satisfied = False
        error_action = None
        error_message = ""

        for i in range(num_actions):

            current_action = actions[i]
            
            # print("Current state", self.block_state)
            # print("Clear state", self.clear_state)
            print("Action", i, ":", current_action)

            with open(test_log_file_path, "a") as f:
                f.write("Action "+ str(i) + ": " + current_action + 
                        ", state: " + np.array2string(self.block_state) +
                        ", clear state: " + np.array2string(self.clear_state) +"\n")


            current_action = current_action.split(' ')
            action_type = current_action[0][1:]

            # execute actions
            if action_type == "unstack":

                is_error, error_message = self.unstack(current_action)

            elif action_type == "stack":

                is_error, error_message = self.stack(current_action)

            elif action_type == "putdown":

                is_error, error_message = self.putdown(current_action)

            elif action_type == "pickup":

                is_error, error_message = self.pickup(current_action)

            else:

                print("Action", action_type, "is not defined.")
                continue

            # if there is error
            if is_error == True:

                error_action = actions[i]

                print("Error:", error_message)
                with open(test_log_file_path, "a") as f:
                    f.write("Error: "+ error_message +"\n")

                break

            # check if goal state is satisfied
            goal_state_trans = self.transform_goal_state(self.goal_state)
            is_goal_satisfied = np.array_equal(goal_state_trans, self.block_state)
            if is_goal_satisfied:

                is_satisfied = True
                with open(test_log_file_path, "a") as f:
                    f.write("Goal state satisfied!" +"\n")
                print("Goal state satisfied!")

                # however, if it is not the last action
                if i < num_actions - 1:

                    is_error = True
                    error_action = actions[i]
                    error_message = "Goal satisfied at " + error_action + ". "
                    with open(test_log_file_path, "a") as f:
                        f.write("Goal satisfied at " + error_action + ". " +"\n")
                    print("Goal satisfied at " + error_action + ". ")

                    break

            # check if action sequence finished but goal is still not satisfied
            elif i == num_actions - 1:

                is_error = True

                for j in range(self.num_blocks):

                    index = self.num_blocks - j - 1
                    # check each goal
                    if j == 0:
                        current_layer_goal = self.goal_state[index] 
                        block_next_layer = 0
                        if self.block_state[current_layer_goal - 1] != block_next_layer:

                            error_message = "Goal b" + str(current_layer_goal) + " on table is not satisfied."
                            break
                    else:
                        current_layer_goal = self.goal_state[index] 
                        block_next_layer = self.goal_state[index+1]
                        if self.block_state[current_layer_goal - 1] != block_next_layer:

                            error_message = "Goal b" + str(current_layer_goal) + " on b" + str(block_next_layer) + " is not satisfied. "
                            break


                # error_message = "Goal is not satisfied. "
                with open(test_log_file_path, "a") as f:
                    f.write("Error: "+ error_message +"\n")
                print(error_message)

        return is_satisfied, is_error, error_message, error_action

    # unstack b1 from b2
    def unstack(self, action):

        is_error = False
        error_message = None

        b1_index = int(action[1][1])
        b2_index = int(action[2][1])

        # check if pre-conditions are satisfied
        # hand is empty
        if self.is_hand_empty == False:

            is_error = True
            block_in_hand = np.where(self.block_state==-1)[0][0] + 1
            error_message = "Hand is not empty when unstacking. Please add putdown b" + str(block_in_hand) + " before this action. "
            return is_error, error_message

        # b1 is clear
        if self.clear_state[b1_index-1] != 1:

            is_error = True
            block_on_b1 = np.where(self.block_state==b1_index)[0][0] + 1
            error_message = "b" + str(b1_index) + " is not clear to move. b" + str(block_on_b1) + " is on top of it. Please add unstack b" + str(block_on_b1) + " from b" + str(b1_index) + " before this action. "
            return is_error, error_message

        # (on b1 b2)
        block_under_b1 = self.block_state[b1_index-1]
        if block_under_b1 != b2_index:

            is_error = True
            if block_under_b1 == 0:
                error_message = "b" + str(b1_index) + " is on the table. Please replace (unstack b" +str(b1_index)  +  " b" + str(b2_index) + ") with (pickup b" + str(b1_index) + "). " 
            else:
                error_message = "b" + str(b1_index) + " is not on top of b" + str(b2_index) + ". "
            return is_error, error_message


        # if no error: execute the action
        self.block_state[b1_index - 1] = -1 # in hand
        self.clear_state = self.determine_clear_state(self.block_state)
        self.is_hand_empty = False 

        return is_error, error_message

    # stack b1 on b2
    def stack(self, action):

        is_error = False
        error_message = None

        b1_index = int(action[1][1])
        b2_index = int(action[2][1])

        # check if pre-conditions are satisfied
        # hand is not empty
        if self.is_hand_empty == True:

            is_error = True
            if self.block_state[b1_index-1] == 0:
                error_message = "Hand is empty when stacking b" + str(b1_index) + ". Please add pickup b" + str(b1_index) + " before this action. " 
            else:
                error_message = "Hand is empty when stacking b" + str(b1_index) + ". Please add unstack b" + str(b1_index) + " b" + str(self.block_state[b1_index-1]) + " before this action. " 
            return is_error, error_message

        # b1 is in hand
        if self.block_state[b1_index-1] != -1:

            is_error = True
            error_message = "b" + str(b1_index) + " is not in hand. "
            return is_error, error_message

        # b2 is clear
        if self.clear_state[b2_index-1] != 1:

            is_error = True
            block_on_b2 = np.where(self.block_state==b2_index)[0][0] + 1
            error_message = "b" + str(b2_index) + " is not clear to move. b" + str(block_on_b2) + " is on top of it. Please add unstack b" + str(block_on_b2) + " from b" + str(b2_index) + " before this action. "
            return is_error, error_message

        # if no error: execute the action
        self.block_state[b1_index - 1] = b2_index # on table
        self.clear_state = self.determine_clear_state(self.block_state)
        self.is_hand_empty = True 

        return is_error, error_message

    # putdown b1
    def putdown(self, action):

        is_error = False
        error_message = None

        b1_index = int(action[1][1])

        # check if pre-conditions are satisfied
        # hand is not empty
        if self.is_hand_empty == True:

            is_error = True
            error_message = "Hand is empty when putting down b" + str(b1_index) + ". "
            return is_error, error_message

        # b1 is in hand
        if self.block_state[b1_index-1] != -1:

            is_error = True
            error_message = "b" + str(b1_index) + " is not in hand. "
            return is_error, error_message

        # if no error: execute the action
        self.block_state[b1_index - 1] = 0 # on table
        self.clear_state = self.determine_clear_state(self.block_state)
        self.is_hand_empty = True 

        return is_error, error_message

    # pickup b1 from table
    def pickup(self, action):

        is_error = False
        error_message = None

        b1_index = int(action[1][1])

        # check if pre-conditions are satisfied
        # hand empty
        if self.is_hand_empty == False:

            is_error = True
            block_in_hand = np.where(self.block_state==-1)[0][0] + 1
            error_message = "Hand is not empty when picking up. Please add putdown b" + str(block_in_hand) + " before this action. "
            return is_error, error_message

        # b1 is clear
        if self.clear_state[b1_index-1] != 1:

            is_error = True
            block_on_b1 = np.where(self.block_state==b1_index)[0][0] + 1
            error_message = "b" + str(b1_index) + " is not clear to move. b" + str(block_on_b1) + " is on top of it. Please add unstack b" + str(block_on_b1) + " from b" + str(b1_index) + " before this action. "
            return is_error, error_message

        # b1 is on table
        if self.block_state[b1_index-1] != 0:

            is_error = True
            error_message = "b" + str(b1_index) + " is on top of b" + str(self.block_state[b1_index-1]) + ". Please replace (pickup b" +str(b1_index)  + ") with (unstack b" + str(b1_index) + " b" + str(self.block_state[b1_index-1]) + "). "
            return is_error, error_message


        # if no error: execute the action
        self.block_state[b1_index - 1] = -1 # in hand
        self.clear_state = self.determine_clear_state(self.block_state)
        self.is_hand_empty = False 

        return is_error, error_message        

    # transform goal state to the same representation of block state for comparision
    def transform_goal_state(self, goal_state):

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