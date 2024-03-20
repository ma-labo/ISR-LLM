import openai
import os
import argparse
import datetime
import numpy as np
import time
import re

from utils.utils import *
from LLM.Translator.Translator import Translator
from LLM.Planner.Planner import Planner
from LLM.Validator.Validator import Validator
from Blocks_Sim.Block_Sim import BlockSim
from BallMoving_Sim.BallMoving_Sim import BallMovingSim
from Cooking_Sim.Cooking_Sim import CookingSim

DOMAINS = ["blocksworld", "ballmoving", "cooking"]
METHODS = ["LLM_trans_self_feedback", "LLM_trans_no_feedback", "LLM_trans_exact_feedback", "LLM_no_trans", "LLM_no_trans_self_feedback"]
MODELS = ["gpt-3.5-turbo","gpt-4", "gpt-3.5-turbo-16k"]

# LLM planning without PDDL translator
def test_LLM_no_trans(test_initial_state, test_goal_state, num_test, max_num_refine,
                                  max_refine_temperature, num_prompt_examples_dataset, 
                                  test_log_file_path, gpt_api_wait_time):


    # test loop
    for i in range(num_test):

        # wait for every loop (gpt api has rpm limit)
        time.sleep(gpt_api_wait_time)

        initial_state = test_initial_state[i + num_prompt_examples_dataset, :]
        goal_state = test_goal_state[i + num_prompt_examples_dataset, :]

        # generate description
        description = scenario_simulator.generate_scene_description(initial_state= initial_state,
                                                        goal_state = goal_state, 
                                                        constraint = None)

        print(i + num_prompt_examples_dataset, description)

        with open(test_log_file_path, "a") as f:
            f.write("Test case index: " + str(i+num_prompt_examples_dataset) + "\n")
            f.write(description +"\n")

        planning_problem = description + '\n'

        # refine loop
        for j in range(max_num_refine + 1):

            time.sleep(gpt_api_wait_time)

            # (re-)initialize block sim
            scenario_simulator.initialize_state(initial_state= initial_state, goal_state = goal_state, constraint = None)

            with open(test_log_file_path, "a") as f:
                f.write("Attempt: " + str(j) + "\n")

            # LLM planner
            # increase temperature if more attempts
            if i == 0: 
                temperature = 0
            else:
                temperature = min(max_refine_temperature, 0.1*i)
            response_planner = LLM_Planner.query(planning_problem, is_append = True, temperature = temperature)
            # print(response_planner)

            # LLM planner
            action_sequence = response_planner["choices"][0]["message"]["content"]

            # simulate actions
            print("Attempt", j)
            print(action_sequence)
            with open(test_log_file_path, "a") as f:
                f.write(action_sequence +"\n")
                f.write("Analysis: " +"\n")

            # validation using external simulation
            is_satisfied, is_error, error_message, error_action = scenario_simulator.simulate_actions(action_sequence, test_log_file_path)

            # if error is found
            if is_error == True:

                if is_satisfied == False:

                    # some action is wrong
                    if error_action != None:

                        error_description = "Action " + error_action + " is wrong. Error info: " + error_message
                        planning_problem = error_description + "Please find a new plan. "

                    # no correct action is returned by gpt response
                    else:

                        error_description = error_message
                        planning_problem = error_description + "Please find a new plan. "

                # if goal is satisfied before action sequence finishes
                else:

                    error_description = error_message + "Please ignore actions after action " + error_action
                    planning_problem = error_description

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            # if no action returned
            elif is_error == None:

                error_description = "Please find a new plan. "
                planning_problem = error_description

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            # if no error found
            else:

                # exit refine loop
                break

        # reinitialize planner for next case
        LLM_Planner.init_messages(is_reinitialize = True)

        with open(test_log_file_path, "a") as f:
            f.write("End of test case " + str(i+num_prompt_examples_dataset) + "\n\n\n")


# LLM planning without PDDL translator and using self feedback
def test_LLM_no_trans_self_feedback(domain, test_initial_state, test_goal_state, num_test, max_num_refine,
                                  max_refine_temperature, num_prompt_examples_dataset, 
                                  test_log_file_path, gpt_api_wait_time):


    # test loop
    for i in range(num_test):

        # wait for every loop (gpt api has rpm limit)
        time.sleep(gpt_api_wait_time)

        initial_state = test_initial_state[i + num_prompt_examples_dataset, :]
        goal_state = test_goal_state[i + num_prompt_examples_dataset, :]

        # generate description
        description = scenario_simulator.generate_scene_description(initial_state= initial_state,
                                                        goal_state = goal_state, 
                                                        constraint = None)

        print(i + num_prompt_examples_dataset, description)

        with open(test_log_file_path, "a") as f:
            f.write("Test case index: " + str(i+num_prompt_examples_dataset) + "\n")
            f.write(description +"\n")

        # LLM translator
        planning_problem = description + '\n'
        # print(response_translator)

        # refine loop
        for j in range(max_num_refine + 1):

            time.sleep(gpt_api_wait_time)

            # (re-)initialize block sim
            scenario_simulator.initialize_state(initial_state= initial_state, goal_state = goal_state, constraint = None)

            with open(test_log_file_path, "a") as f:
                f.write("Attempt: " + str(j) + "\n")

            # LLM planner
            if i == 0: 
                temperature = 0
            else:
                temperature = min(max_refine_temperature, 0.1*i)
            response_planner = LLM_Planner.query(planning_problem, is_append = True, temperature = temperature)
            # print(response_planner)

            # LLM planner
            action_sequence = response_planner["choices"][0]["message"]["content"]

            # simulate actions
            print("Attempt", j)
            print(action_sequence)
            with open(test_log_file_path, "a") as f:
                f.write(action_sequence +"\n")
                f.write("Analysis: " +"\n")

            # self-evaluate actions
            action_description = extract_action_description(action_sequence, domain)

            if domain == 'blocksworld':
                validate_question = "Question:\n" + planning_problem + "Examined action sequence:\n" + action_description
            elif domain == 'ballmoving':
                validate_question = "Question:\n" + planning_problem + "Examined action sequence:\n" + action_description
            elif domain == 'cooking':
                validate_question = "Question:\n" + planning_problem + "Examined action sequence:\n" + action_description

            print(validate_question)
            with open(test_log_file_path, "a") as f:
                f.write(validate_question +"\n")

            time.sleep(gpt_api_wait_time)

            response_validator = LLM_Validator.query(validate_question, is_append=False)
            response_validator_content = response_validator["choices"][0]["message"]["content"]
            with open(test_log_file_path, "a") as f:
                f.write(response_validator_content +"\n")

            # Answer
            valid_result = response_validator_content.split('Final answer:', 1)
            if len(valid_result) == 1: # no action returned
                break
                valid_result = 'No'
            else:
                valid_result = valid_result[1]

            if 'Yes' in valid_result:

                print("Self-evaluation suggests a solution.")
                with open(test_log_file_path, "a") as f:
                    f.write("Self-evaluation suggests a solution.\n")
                # exit self-refine loop
                break

            elif 'No' in valid_result:

                print("Self-evaluation suggests a failure.")
                error_description = "Goal is not satisfied." #Error analysis:" + summary_content 
                if domain == 'blocksworld':
                    planning_problem = error_description + " Please find a new plan by considering the goals from bottom to top. "
                elif domain == 'ballmoving':
                    planning_problem = error_description + " Please find a new plan by considering the locations of balls. "
                elif domain == 'cooking':
                    planning_problem = error_description + " Please find a new plan by considering the ingredients needed in each pot. "
                print(planning_problem)

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            else:

                print("Unknown results:", valid_result)

        # check actual results
        print("Actual analysis:")
        with open(test_log_file_path, "a") as f:
            f.write("Actual analysis:\n")
        is_satisfied, is_error, error_message, error_action = scenario_simulator.simulate_actions(action_sequence, test_log_file_path)

        # reinitialize planner for next case
        LLM_Planner.init_messages(is_reinitialize = True)

        with open(test_log_file_path, "a") as f:
            f.write("End of test case " + str(i+num_prompt_examples_dataset) + "\n\n\n")

# LLM planning with PDDL translator and using exact feedback from external validator
def test_LLM_trans_exact_feedback(test_initial_state, test_goal_state, num_test, max_num_refine,
                                  max_refine_temperature, num_prompt_examples_dataset, 
                                  test_log_file_path, gpt_api_wait_time):


    # test loop
    for i in range(num_test):

        # wait for every loop (gpt api has rpm limit)
        time.sleep(gpt_api_wait_time)

        initial_state = test_initial_state[i + num_prompt_examples_dataset, :]
        goal_state = test_goal_state[i + num_prompt_examples_dataset, :]

        # generate description
        description = scenario_simulator.generate_scene_description(initial_state= initial_state,
                                                        goal_state = goal_state, 
                                                        constraint = None)

        print(i + num_prompt_examples_dataset, description)

        with open(test_log_file_path, "a") as f:
            f.write("Test case index: " + str(i+num_prompt_examples_dataset) + "\n")
            f.write(description +"\n")

        # LLM translator
        response_translator = LLM_Translator.query(description, is_append = False)
        planning_problem = response_translator["choices"][0]["message"]["content"]
        # print(response_translator)

        # refine loop
        for j in range(max_num_refine + 1):

            time.sleep(gpt_api_wait_time)

            # (re-)initialize block sim
            scenario_simulator.initialize_state(initial_state= initial_state, goal_state = goal_state, constraint = None)

            with open(test_log_file_path, "a") as f:
                f.write("Attempt: " + str(j) + "\n")

            # LLM planner
            # increase temperature if more attempts
            if i == 0: 
                temperature = 0
            else:
                temperature = min(max_refine_temperature, 0.1*i)
            response_planner = LLM_Planner.query(planning_problem, is_append = True, temperature = temperature)
            # print(response_planner)

            # LLM planner
            action_sequence = response_planner["choices"][0]["message"]["content"]

            # simulate actions
            print("Attempt", j)
            print(action_sequence)
            with open(test_log_file_path, "a") as f:
                f.write(action_sequence +"\n")
                f.write("Analysis: " +"\n")

            # validation using external simulation
            is_satisfied, is_error, error_message, error_action = scenario_simulator.simulate_actions(action_sequence, test_log_file_path)

            # if error is found
            if is_error == True:

                if is_satisfied == False:

                    # some action is wrong
                    if error_action != None:

                        error_description = "Action " + error_action + " is wrong. Error info: " + error_message
                        planning_problem = error_description + "Please find a new plan. "

                    # no correct action is returned by gpt response
                    else:

                        error_description = error_message
                        planning_problem = error_description + "Please find a new plan. "

                # if goal is satisfied before action sequence finishes
                else:

                    error_description = error_message + "Please ignore actions after action " + error_action
                    planning_problem = error_description

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            # if no action returned
            elif is_error == None:

                error_description = "Please find a new plan. "
                planning_problem = error_description

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            # if no error found
            else:

                # exit refine loop
                break

        # reinitialize planner for next case
        LLM_Planner.init_messages(is_reinitialize = True)

        with open(test_log_file_path, "a") as f:
            f.write("End of test case " + str(i+num_prompt_examples_dataset) + "\n\n\n")

# LLM planning with PDDL translator and using self feedback
def test_LLM_trans_self_feedback(domain, test_initial_state, test_goal_state, num_test, max_num_refine,
                                  max_refine_temperature, num_prompt_examples_dataset, 
                                  test_log_file_path, gpt_api_wait_time):


    # test loop
    for i in range(num_test):

        # wait for every loop (gpt api has rpm limit)
        time.sleep(gpt_api_wait_time)

        initial_state = test_initial_state[i + num_prompt_examples_dataset, :]
        goal_state = test_goal_state[i + num_prompt_examples_dataset, :]

        # generate description
        description = scenario_simulator.generate_scene_description(initial_state= initial_state,
                                                        goal_state = goal_state, 
                                                        constraint = None)

        print(i + num_prompt_examples_dataset, description)

        with open(test_log_file_path, "a") as f:
            f.write("Test case index: " + str(i+num_prompt_examples_dataset) + "\n")
            f.write(description +"\n")

        # LLM translator
        response_translator = LLM_Translator.query(description, is_append = False)
        planning_problem = response_translator["choices"][0]["message"]["content"]
        # print(response_translator)

        # initial and goal state in pddl
        pddl_init_state, pddl_goal_state = extract_state_pddl(planning_problem, domain)

        # refine loop
        for j in range(max_num_refine + 1):

            time.sleep(gpt_api_wait_time)

            # (re-)initialize block sim
            scenario_simulator.initialize_state(initial_state= initial_state, goal_state = goal_state, constraint = None)

            with open(test_log_file_path, "a") as f:
                f.write("Attempt: " + str(j) + "\n")

            # LLM planner
            if i == 0: 
                temperature = 0
            else:
                temperature = min(max_refine_temperature, 0.1*i)
            response_planner = LLM_Planner.query(planning_problem, is_append = True, temperature = temperature)
            # print(response_planner)

            # LLM planner
            action_sequence = response_planner["choices"][0]["message"]["content"]

            # simulate actions
            print("Attempt", j)
            print(action_sequence)
            with open(test_log_file_path, "a") as f:
                f.write(action_sequence +"\n")
                f.write("Analysis: " +"\n")

            # self-evaluate actions
            action_description = extract_action_description(action_sequence, domain)

            if domain == 'blocksworld':
                validate_question = "Question:\nBlock initial state:\n" + pddl_init_state + "\nGoal state:\n" + pddl_goal_state + "\nExamined action sequence:\n" + action_description
            elif domain == 'ballmoving':
                validate_question = "Question:\nRobot and ball initial state: \n" + pddl_init_state + "\nGoal state:\n" + pddl_goal_state + "\nExamined action sequence:\n" + action_description
            elif domain == 'cooking':
                validate_question = "Question:\nInitial state: \n" + pddl_init_state + "\nGoal state:\n" + pddl_goal_state + "\nExamined action sequence:\n" + action_description

            print(validate_question)
            with open(test_log_file_path, "a") as f:
                f.write(validate_question +"\n")

            time.sleep(gpt_api_wait_time)

            response_validator = LLM_Validator.query(validate_question, is_append=False)
            response_validator_content = response_validator["choices"][0]["message"]["content"]
            with open(test_log_file_path, "a") as f:
                f.write(response_validator_content +"\n")

            # Answer
            valid_result = response_validator_content.split('Final answer:', 1)
            if len(valid_result) == 1: # no action returned
                break
                valid_result = 'No'
            else:
                valid_result = valid_result[1]

            if 'Yes' in valid_result:

                print("Self-evaluation suggests a solution.")
                with open(test_log_file_path, "a") as f:
                    f.write("Self-evaluation suggests a solution.\n")
                # exit self-refine loop
                break

            elif 'No' in valid_result:

                print("Self-evaluation suggests a failure.")
                error_description = "Goal is not satisfied." #Error analysis:" + summary_content 
                if domain == 'blocksworld':
                    planning_problem = error_description + " Please find a new plan by considering the goals from bottom to top. "
                elif domain == 'ballmoving':
                    planning_problem = error_description + " Please find a new plan by considering the locations of balls. "
                elif domain == 'cooking':
                    planning_problem = error_description + " Please find a new plan by considering the ingredients needed in each pot. "
                print(planning_problem)

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            else:

                print("Unknown results:", valid_result)

        # check actual results
        print("Actual analysis:")
        with open(test_log_file_path, "a") as f:
            f.write("Actual analysis:\n")
        is_satisfied, is_error, error_message, error_action = scenario_simulator.simulate_actions(action_sequence, test_log_file_path)

        # reinitialize planner for next case
        LLM_Planner.init_messages(is_reinitialize = True)

        with open(test_log_file_path, "a") as f:
            f.write("End of test case " + str(i+num_prompt_examples_dataset) + "\n\n\n")


if __name__=="__main__":

    # Openai Key (please replace with your own key)
    # openai.api_key = 'YOUR-KEY'
    openai.api_key = 'YOUR-KEY'

    # parameters
    parser = argparse.ArgumentParser(description="LLM-Task-Planner")
    parser.add_argument('--domain', type=str, choices=DOMAINS, default="blocksworld")
    parser.add_argument('--method', type=str, choices=METHODS, default="LLM_no_trans")
    parser.add_argument('--model', type=str, choices=MODELS, default="gpt-3.5-turbo")
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--num_objects', type=int, choices=[3,4], default=3)
    parser.add_argument('--num_trans_example', type=int, choices=[1,2,3], default=3)
    parser.add_argument('--num_plan_example', type=int, choices=[3,4,5], default=4)
    parser.add_argument('--num_valid_example', type=int, choices=[4,5,6], default=6)
    args = parser.parse_args()

    # initialize log dir
    if args.logdir == None:
        args.logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_log/")
        args.logdir = args.logdir + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)

    # Initialize translator
    LLM_Translator = Translator(args, is_log_example=True)

    # Initialize planner
    LLM_Planner = Planner(args, is_log_example=True)

    # Initialize validator 
    if args.method == "LLM_trans_self_feedback" or args.method == "LLM_no_trans_self_feedback":
        LLM_Validator = Validator(args, is_log_example=True)

    # Initialize block simulator
    if args.domain == 'blocksworld':
        scenario_simulator = BlockSim()

    elif args.domain == 'ballmoving':
        scenario_simulator = BallMovingSim()

    elif args.domain == 'cooking':
        scenario_simulator = CookingSim()


    ###############################################
    # load test scenarios
    test_initial_state, test_goal_state = load_test_scenarios(args)

    # run test
    num_test = 10
    num_prompt_examples_dataset = 3 # the first n examples are in the prompt example, so skip them
    max_num_refine = 10  # max number of refinement, if it is 0 -> no feedback
    gpt_api_wait_time = 20 # wait time during gpt api call for preventing overload
    max_refine_temperature = 0.4 # maximal value of refine temperature

    # test log
    test_log_file_path = args.logdir + "/test_log.txt"
    with open(test_log_file_path, "w") as f:
        if args.domain == "blocksworld":
            f.write("Test log for "+ args.domain + " with " + str(args.num_objects) + " blocks." +"\n")
        elif args.domain == "ballmoving":
            f.write("Test log for "+ args.domain + " with " + str(args.num_objects) + " balls." +"\n")
        elif args.domain == "cooking":
            f.write("Test log for "+ args.domain + " with " + str(args.num_objects) + " pots." +"\n")


    if args.method == "LLM_trans_no_feedback":

        # if no feedback, use the output directly, same method but no feedback is generated
        max_num_refine = 0
        test_LLM_trans_exact_feedback(test_initial_state, test_goal_state, num_test, max_num_refine, 
                                max_refine_temperature, num_prompt_examples_dataset, test_log_file_path, gpt_api_wait_time)

    elif args.method == "LLM_trans_exact_feedback":

        test_LLM_trans_exact_feedback(test_initial_state, test_goal_state, num_test, max_num_refine, 
                                max_refine_temperature, num_prompt_examples_dataset, test_log_file_path, gpt_api_wait_time)

    elif args.method == "LLM_trans_self_feedback":

        test_LLM_trans_self_feedback(args.domain, test_initial_state, test_goal_state, num_test, max_num_refine, 
                                max_refine_temperature, num_prompt_examples_dataset, test_log_file_path, gpt_api_wait_time)

    elif args.method == "LLM_no_trans":

        max_num_refine = 10
        test_LLM_no_trans(test_initial_state, test_goal_state, num_test, max_num_refine, 
                                max_refine_temperature, num_prompt_examples_dataset, test_log_file_path, gpt_api_wait_time)

    elif args.method == "LLM_no_trans_self_feedback":

        test_LLM_no_trans_self_feedback(args.domain, test_initial_state, test_goal_state, num_test, max_num_refine, 
                                max_refine_temperature, num_prompt_examples_dataset, test_log_file_path, gpt_api_wait_time)

    else:

        raise ValueError("Method not implemented.")





