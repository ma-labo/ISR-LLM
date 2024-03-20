import os
import numpy as np
import re

class BallMovingSim(object):
    """
    Simple ball moving simulator
    """
    def __init__(self):
        
        self.object_state = None
        self.goal_state = None
        self.complete_state = None
        self.constraint = None
        self.is_hand_empty = None

    # Generate a given specific block scenario
    def generate_scene_description(self, initial_state, goal_state, constraint = None):

        num_balls = len(initial_state) - 1

        print("Generating a given "+str(num_balls)+" balls scenario...")
        print("Initial state: ", initial_state)
        print("Goal state: ", goal_state)
        print("Constraint: ", constraint)

        # Description for LLM
        description = self.compose_description(num_balls, initial_state, goal_state, constraint)

        return description

    # Compose task description 
    def compose_description(self, num_balls, initial_state, goal_state, constraint):

        
        # Initial openning
        text_open = "I have " + str(num_balls) + " balls within 4 rooms."
        
        # Initial state
        text_initial_state = " Initially: "
        for i in range(num_balls + 1):

            # robot
            if i == 0:

                text_single = "Robot is in room" + str(initial_state[i])+". "

            else: 

                text_single = "Ball ball" + str(i) + " is in room" + str(initial_state[i])+". "

            text_initial_state = text_initial_state + text_single

        # Condition
        text_condition = ""

        # Goal state
        text_goal = "Your goal is to move the balls to specific rooms: "
        for i in range(num_balls):

            if i < num_balls - 1:

                text_single = "ball" + str(i+1) + " in room"+ str(goal_state[i]) + ", "

            else:

                text_single = "and ball" + str(i+1) + " in room"+ str(goal_state[i]) + ". "


            text_goal = text_goal + text_single

        # Constraint
        text_constraint = ""
        if constraint != None:
            pass

        description = text_open + text_initial_state + text_condition + text_goal + text_constraint

        return description

    # Initialize simulation state
    def initialize_state(self, initial_state, goal_state, constraint):

        self.object_state = initial_state.copy() # use copy for avoiding reference
        self.goal_state = goal_state
        self.constraint = constraint
        self.num_balls = len(initial_state) - 1
        self.is_hand_empty = True        
        self.complete_state = self.determine_complete_state(self.object_state, self.goal_state)

    # determine if ball is already in its goal room
    def determine_complete_state(self, object_state, goal_state):

        complete_state = np.equal(object_state[1:], goal_state)

        return complete_state

    # find the next ball that should be moved according to desired conditions
    def find_next_desired_goal(self):

        robot_room = self.object_state[0]

        # check if in the same room there are unsatisfied balls
        same_room_ball = np.where(self.object_state[1:]==robot_room)[0]
        # no ball in the same room
        if len(same_room_ball) == 0:

            # goal is the first unsatisfied ball
            goal_actual = np.where(self.complete_state==False)[0]
            if len(goal_actual) == 0:
                print("all balls are satisfied, no next goal")
                goal_actual = None
            else:
                goal_actual = goal_actual[0] + 1

        # if there are balls in the same room
        else:

            goal_actual = None
            for j in range(len(same_room_ball)):
                # pick first unsatisfied ball if possible
                if self.complete_state[same_room_ball[j]] == False:
                    goal_actual = same_room_ball[j]+1
                    break
            # all ball in same room is satisfied, then first unsatisfied ball
            if goal_actual == None:
                goal_actual = np.where(self.complete_state==False)[0]
                if len(goal_actual) == 0:
                    print("all balls are satisfied, no next goal")
                    goal_actual = None
                else:
                    goal_actual = goal_actual[0] + 1

        return goal_actual

    # Simulate given action sequence
    def simulate_actions(self, action_sequence, test_log_file_path):

        # extract valid actions
        actions = re.findall(r'\(.*?\)', action_sequence)
        filtered_actions = []
        for i in range(len(actions)):
            current_action = actions[i]
            current_action = current_action.split(' ')
            action_type = current_action[0][1:]
            if action_type == 'pick' or action_type == 'move' or action_type == 'drop':
                filtered_actions.append(actions[i])
        print(filtered_actions)
        num_actions = len(filtered_actions)

        is_error = None
        is_satisfied = False
        error_action = None
        error_message = ""

        for i in range(num_actions):

            current_action = filtered_actions[i]
            
            print("Current state", self.object_state)
            print("Complete state", self.complete_state)
            print("Action", i, ":", current_action)

            with open(test_log_file_path, "a") as f:
                f.write("Action "+ str(i) + ": " + current_action + 
                        ", state: " + np.array2string(self.object_state) +
                        ", complete state: " + np.array2string(self.complete_state) +"\n")


            current_action = current_action.split(' ')
            action_type = current_action[0][1:]

            # execute actions
            if action_type == "pick":

                is_error, error_message = self.pick(current_action)

            elif action_type == "drop":

                is_error, error_message = self.drop(current_action)

            elif action_type == "move":

                is_error, error_message = self.move(current_action)

            else:

                print("Action", action_type, "is not defined.")
                continue

            # if there is error
            if is_error == True:

                error_action = filtered_actions[i]

                print("Error:", error_message)
                with open(test_log_file_path, "a") as f:
                    f.write("Error: "+ error_message +"\n")

                break

            # check if goal state is satisfied
            is_goal_satisfied = np.array_equal(self.goal_state, self.object_state[1:])
            if is_goal_satisfied:

                is_satisfied = True
                with open(test_log_file_path, "a") as f:
                    f.write("Goal state satisfied!" +"\n")
                print("Goal state satisfied!")

                # however, if it is not the last action
                if i < num_actions - 1:

                    is_error = True
                    error_action = filtered_actions[i]
                    error_message = "Goal satisfied at " + error_action + ". "
                    with open(test_log_file_path, "a") as f:
                        f.write("Goal satisfied at " + error_action + ". " +"\n")
                    print("Goal satisfied at " + error_action + ". ")

                    break

            # check if action sequence finished but goal is still not satisfied
            elif i == num_actions - 1:

                is_error = True

                for j in range(self.num_balls):

                    if self.complete_state[j] == False:
                        error_message = "ball" + str(j+1) + " is not satisfied. "
                        break

                with open(test_log_file_path, "a") as f:
                    f.write("Error: "+ error_message +"\n")
                print(error_message)

        return is_satisfied, is_error, error_message, error_action

    # pick ball room
    def pick(self, action):

        is_error = False
        error_message = None

        ball_index = int(action[1][4])
        room_index = int(action[2][4])

        # check if pre-conditions are satisfied
        # hand is empty
        if self.is_hand_empty == False:

            is_error = True
            ball_in_hand = np.where(self.object_state==-1)[0][0] 
            error_message = "Hand is not empty when picking. Please add (drop ball" + str(ball_in_hand) + ") before this action. "
            return is_error, error_message

        # robot is in the room
        if self.object_state[0] != room_index:

            is_error = True
            error_message = "robot1 is not in room" + str(room_index) +". Please add (move robot1 room" + str(self.object_state[0]) + " room" + str(room_index) + ") before this action. "
            return is_error, error_message

        # ball is in the room
        if self.object_state[ball_index] != room_index:

            is_error = True
            error_message = "ball" + str(ball_index) +" is not in room" + str(room_index) +". Please add (move robot1 room" + str(self.object_state[0]) + " room" + str(self.object_state[ball_index]) + ") before this action. "
            return is_error, error_message

        # if no error: execute the action
        self.object_state[ball_index] = -1 # in hand
        self.complete_state = self.determine_complete_state(self.object_state, self.goal_state)
        self.is_hand_empty = False 

        return is_error, error_message

    # drop ball room
    def drop(self, action):

        is_error = False
        error_message = None

        ball_index = int(action[1][4])
        room_index = int(action[2][4])

        # check if pre-conditions are satisfied
        # hand is not empty
        if self.is_hand_empty == True:

            is_error = True
            error_message = "Hand is empty when droping. Please pick ball" + str(ball_index) + " before this action. "
            return is_error, error_message

        # ball is in hand
        if self.object_state[ball_index] != -1:

            is_error = True
            error_message = "ball" + str(ball_index) +" is not in hand. Please pick ball" + str(ball_index) + " before this action. "
            return is_error, error_message

        # robot is in the room
        if self.object_state[0] != room_index:

            is_error = True
            error_message = "robot1 is not in room" + str(room_index) +". Please add (move robot1 room" + str(self.object_state[0]) + " room" + str(room_index) + ") before this action. "
            return is_error, error_message

        # if no error: execute the action
        self.object_state[ball_index] = room_index 
        self.complete_state = self.determine_complete_state(self.object_state, self.goal_state)
        self.is_hand_empty = True 

        return is_error, error_message

    # putdown robot1 room1 room2
    def move(self, action):

        is_error = False
        error_message = None

        from_room_index = int(action[2][4])
        to_room_index = int(action[3][4])

        # check if pre-conditions are satisfied
        # robot is in the room
        if self.object_state[0] != from_room_index:

            is_error = True
            error_message = "robot1 is not in room" + str(from_room_index) +". Please add (move robot1 room" + str(self.object_state[0]) + " room" + str(from_room_index) + ") before this action. "
            return is_error, error_message

        # if no error: execute the action
        self.object_state[0] = to_room_index
        self.complete_state = self.determine_complete_state(self.object_state, self.goal_state)

        return is_error, error_message