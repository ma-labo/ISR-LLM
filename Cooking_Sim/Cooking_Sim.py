import os
import numpy as np
import re

class CookingSim(object):
    """
    Simple cooking simulator
    """
    def __init__(self):
        
        self.pot_state = None
        self.ingredient_picked_state = None
        self.goal_state = None
        self.constraint = None
        self.is_hand_empty = None
        self.num_pots = None
        self.ingredient_in_hand = None
        # number of default ingredients
        self.num_ingredients = 6

    # Generate a given specific block scenario
    def generate_scene_description(self, initial_state, goal_state, constraint = None):

        num_pots = int(len(goal_state)/self.num_ingredients)

        print("Generating a given "+str(num_pots)+" pots scenario...")
        print("Initial state: ", initial_state.reshape((num_pots, self.num_ingredients)).tolist())
        print("Goal state: ", goal_state.reshape((num_pots, self.num_ingredients)).tolist())
        print("Constraint: ", constraint)

        # Description for LLM
        description = self.compose_description(num_pots, initial_state, goal_state, constraint)

        return description

    # Compose task description 
    def compose_description(self, num_pots, initial_state, goal_state, constraint):

        
        goal_state = goal_state.reshape((num_pots, self.num_ingredients))

        # Initial openning
        text_open = "I have " + str(num_pots) + " pots and 6 different ingredients."

        # Condition
        text_condition = "Each ingredient can only be picked up once. "

        # Goal state
        text_goal = "Your goal is to add ingredients to pots by following the receipts: "
        for i in range(num_pots):

            ingre_index = np.where(goal_state[i,:]==1)[0]
            n_ingre = len(ingre_index)

            text_single = "pot" + str(i+1) + " contains "

            for j in range(n_ingre):

                text_single = text_single + "ingredient" + str(ingre_index[j]+1)
                if j < n_ingre - 1:
                    text_single = text_single + ", "
                else:
                    text_single = text_single + ". "

            text_goal = text_goal + text_single

        # Constraint
        text_constraint = ""
        if constraint != None:
            text_constraint = "However: "
            for i in range(len(constraint)):

                constraint_single = constraint[i]
                text_single = "b" + str(constraint_single[0]) + " should not be on top of b" + str(constraint_single[1]) + " in all time. "
                text_constraint = text_constraint + text_single

        description = text_open + text_condition + text_goal + text_constraint

        return description

    # Initialize simulation state
    def initialize_state(self, initial_state, goal_state, constraint):

        self.num_pots = int(len(goal_state)/self.num_ingredients)
        self.pot_state = initial_state.copy() # use copy for avoiding reference
        self.pot_state = self.pot_state.reshape((self.num_pots, self.num_ingredients))
        self.goal_state = goal_state.copy()
        self.goal_state = self.goal_state.reshape((self.num_pots, self.num_ingredients))
        
        self.constraint = constraint
        self.is_hand_empty = True
        self.ingredient_in_hand = None

        # if ingredient has been picked already: 0 means not yet, 1 means yes
        self.ingredient_picked_state = np.zeros(self.num_ingredients, dtype=int)


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
            
            print("Action", i, ":", current_action)

            with open(test_log_file_path, "a") as f:
                f.write("Action "+ str(i) + ": " + current_action + 
                        ", current pot state: " + str(self.pot_state.tolist()) +
                        ", ingredient picked state: " + np.array2string(self.ingredient_picked_state) +"\n")


            current_action = current_action.split(' ')
            action_type = current_action[0][1:]

            # execute actions
            if action_type == "pick":

                is_error, error_message = self.pick(current_action)

            elif action_type == "putdown":

                is_error, error_message = self.putdown(current_action)

            elif action_type == "add":

                is_error, error_message = self.add(current_action)

            else:

                print("Action", action_type, "is not defined.")
                continue
                #raise ValueError("Action", action_type, "is not defined.")

            # if there is error
            if is_error == True:

                error_action = actions[i]

                print("Error:", error_message)
                with open(test_log_file_path, "a") as f:
                    f.write("Error: "+ error_message +"\n")

                break

            # check if goal state is satisfied in the last action
            if i == num_actions - 1:

                is_goal_satisfied = np.array_equal(self.goal_state, self.pot_state)
                print(is_goal_satisfied, self.goal_state, self.pot_state)
                if is_goal_satisfied:

                    is_satisfied = True
                    with open(test_log_file_path, "a") as f:
                        f.write("Goal state satisfied!" +"\n")
                    print("Goal state satisfied!")

                else:

                    is_error = True
                    # check each pot
                    for j in range(self.num_pots):
                        for k in range(self.num_ingredients):

                            pot = self.pot_state[j,k]
                            goal = self.goal_state[j,k]

                            if pot != goal:

                                if pot == 1:
                                    error_message = "Goal is not satisfied. pot" + str(j+1) + " should not contain ingredient" + str(k+1) + ". "
                                elif pot == 0:
                                    error_message = "Goal is not satisfied. pot" + str(j+1) + " should contain ingredient" + str(k+1) + ". "

                                break

                    with open(test_log_file_path, "a") as f:
                        f.write("Error: "+ error_message +"\n")
                    print(error_message)

        return is_satisfied, is_error, error_message, error_action

    # pick ingredient
    def pick(self, action):

        is_error = False
        error_message = None

        ingre_index = int(action[1][10])

        # check if pre-conditions are satisfied
        # hand is empty
        if self.is_hand_empty == False:

            is_error = True
            ingre_in_hand = self.ingredient_in_hand
            error_message = "Hand is not empty when picking. Please first putdown ingredient" + str(ingre_in_hand) + " before this action. "
            return is_error, error_message

        # ingredient has not been picked yet
        if self.ingredient_picked_state[ingre_index-1] != 0:

            is_error = True
            error_message = "ingredient" + str(ingre_index) + " has already been picked. "
            return is_error, error_message

        # if no error: execute the action
        self.is_hand_empty = False 
        self.ingredient_in_hand = ingre_index
        self.ingredient_picked_state[ingre_index-1] = 1

        return is_error, error_message

    # putdown ingredient
    def putdown(self, action):

        is_error = False
        error_message = None

        ingre_index = int(action[1][10])

        # check if pre-conditions are satisfied
        # hand is not empty
        if self.is_hand_empty == True:

            is_error = True
            error_message = "Hand is empty when putting down ingredient" + str(ingre_index) + ". "
            return is_error, error_message

        # if no error: execute the action
        self.is_hand_empty = True 
        self.ingredient_in_hand = None

        return is_error, error_message

    # add ingredient to pot
    def add(self, action):

        is_error = False
        error_message = None

        ingre_index = int(action[1][10])
        pot_index = int(action[2][3])

        # check if pre-conditions are satisfied
        # hand is not empty
        if self.is_hand_empty == True:

            is_error = True
            error_message = "Hand is empty when adding ingredient" + str(ingre_index) + ". "
            return is_error, error_message

        # correct ingredient is in hand
        if self.ingredient_in_hand != ingre_index:

            is_error = True
            error_message = "ingredient" + str(ingre_index) + " is not in hand. "
            return is_error, error_message

        # if no error: execute the action
        self.pot_state[pot_index-1, ingre_index - 1] = 1

        return is_error, error_message        