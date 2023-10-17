import json
import re
from abc import ABC, abstractmethod

from utils import query_api, load_openai_cache
import textwrap
from sklearn.metrics import roc_auc_score


class BaseActiveLearningAgent(ABC):
    
    def __init__(self, target_specification_file, engine, openai_cache_file=None, **kwargs):
        self.get_gold_domain_info(target_specification_file)
        self.engine = engine
        self.openai_cache_file = openai_cache_file
        self.openai_cache = load_openai_cache(openai_cache_file)
        self.temperature = kwargs.get("temperature", 0.0)

        self.interaction_history = []


    def get_gold_domain_info(self, target_specification_file):
        '''Gets the gold domain specification that the model should try to learn and other associated information.
        '''
        gold_task = json.load(open(target_specification_file))  #"sample_tests.json"
        for key in gold_task:
            setattr(self, key, gold_task[key])
            if key == "regex":
                self.gold_regex_text = self.regex
                self.gold_regex = re.compile(self.gold_regex_text)
        self.persona_text = self.persona

    def get_task_description(self):
        return "validate an email address adheres to a specific format"

    @staticmethod
    def format_questions_and_answers(questions_and_answers):
        '''Formats the questions and answers into a string.

        Looks like:
        - Should the system allow numbers in the domain? -> Yes

        Args:
            questions_and_answers (list): A list of tuples of the form (question, answer).
        
        Returns:
            str: The formatted questions and answers.
        '''
        return '\n'.join([f"- {question} -> {answer}" for question, answer in questions_and_answers])

    def get_test_case_prompt(self, interaction_history, test_case):
        hypothesis_prompt = textwrap.dedent('''\
            {single_instance_prompt1}
            {previous_examples}
            
            {single_instance_prompt2}
            {test_case}
            '''
        ).format(
            single_instance_prompt1=self.test_case_prompt[0],
            previous_examples=self.format_questions_and_answers(interaction_history),
            single_instance_prompt2=self.test_case_prompt[1],
            test_case=test_case,
        )
        return [{"role": "user", "content": hypothesis_prompt}]
    
    def generate_test_case_answer(self, test_case):
        test_case_messages = self.get_test_case_prompt(self.interaction_history, test_case)
        test_case_answer, _ = query_api(test_case_messages, self.engine, self.openai_cache, self.openai_cache_file)
        test_case_answer = test_case_answer.strip().lower()
        
        return test_case_answer

    def score_test_cases_direct(self, start_metrics=None):
        """
        Condition on query answers directly to score the test cases.
        start_metrics (dict): metrics at the start of the interaction, set to None if computing absolute metrics, else compute relative metrics
            
        Returns:
        Tuple[Dict, List[Dict]]: A tuple of the following:
            Dict: scores (dict): A dictionary containing the accuracy and F1 score of the answers on the test cases.
                accuracy (float): The accuracy of the answers on the test cases.
                AUCROC (float): The AUCROC score of the answers on the test cases.
                correct_prob (float): The probability on the correct answer.
            List[Dict]: all_test_details (list): A list of dictionaries containing the details of each test case.
        """
        # Query Asynchronous API
        all_test_case_messages = []
        test_case_to_answer = {}
        for test_case in self.test_cases:
            # test_case: tuple of (query, answer)
            test_case_messages = self.get_test_case_prompt(self.interaction_history, test_case[0])
            all_test_case_messages.append(test_case_messages)
            answer, _ = query_api(test_case_messages, self.engine, self.openai_cache, self.openai_cache_file)
            test_case_to_answer[json.dumps(test_case_messages)] = answer.strip().lower()

        # Compute Accuracy and AUCROC and correct_prob
        tests_passed = []
        all_test_details = []
        pred_probs = []
        correct_probs = []
        for test_case_message, test_case in zip(all_test_case_messages, self.test_cases):
            while True:
                try:
                    pred_prob = float(test_case_to_answer[json.dumps(test_case_message)].strip().lower())
                    break
                except:
                    test_case_message.append({'role': 'user', 'content': 'Please make your best guess as to a probability. Output the probability and nothing else.'})
                    pred_prob, _ = query_api(test_case_message, self.engine, self.openai_cache, self.openai_cache_file)
                    test_case_to_answer[json.dumps(test_case_message)] = pred_prob
            pred_probs.append(pred_prob)
            pred_answer = 1 if pred_prob > 0.5 else 0
            actual_answer = 1 if test_case[1] else 0
            tests_passed.append(pred_answer == actual_answer)
            correct_probs.append(pred_prob if actual_answer else 1 - pred_prob)
            all_test_details.append({
                "query": test_case[0],
                "pred_prob": pred_prob,
                "pred": pred_answer,
                "actual": actual_answer,
                "correct?": pred_answer == actual_answer,
                "correct_prob": pred_prob if actual_answer else 1 - pred_prob,
            })
        try:
            aucroc = roc_auc_score([test_case[1] for test_case in self.test_cases], pred_probs)
        except:
            # only 1 class present....
            aucroc = 0
        print("====")

        metrics_dict = {
            "accuracy": sum(tests_passed) / len(tests_passed),
            "AUCROC": aucroc,
            "correct_prob": sum(correct_probs) / len(correct_probs),
        }
        if start_metrics is None:
            start_metrics = {
                "accuracy": [metrics_dict["accuracy"]],
                "AUCROC": [metrics_dict["AUCROC"]],
                "correct_prob": [metrics_dict["correct_prob"]],
            }
        metrics_dict["accuracy_relative"] = metrics_dict["accuracy"] - start_metrics["accuracy"][0]
        metrics_dict["AUCROC_relative"] = metrics_dict["AUCROC"] - start_metrics["AUCROC"][0]
        metrics_dict["correct_prob_relative"] = metrics_dict["correct_prob"] - start_metrics["correct_prob"][0]
        
        return metrics_dict, all_test_details

    def score_test_cases(self, start_metrics=None):
        """
        Scores the test cases.

        Args:
            score_type (str): The type of scoring to use. Can be "no_hypothesis", "hypothesis", or "select".

        Returns:
        Tuple[Dict, List[Dict]]: A tuple of the following:
            Dict: scores (dict): A dictionary containing the accuracy and F1 score of the answers on the test cases.
                accuracy (float): The accuracy of the answers on the test cases.
                f1 (float): The F1 score of the answers on the test cases.
            List[Dict]: all_test_details (list): A list of dictionaries containing the details of each test case.
        """
        return self.score_test_cases_direct(start_metrics=start_metrics)

    def generate_hypothesis_regex(self):
        """
        Generates a hypothesis regex given a task description and the previous interaction history.

        Loops until a compileable regex is produced. Regexes that fail to compile are stored in broken_regexes and used to prompt the model for a regex that compiles.
            
        Returns:
            hypothesis_regex (str)
        """
        broken_regexes = []

        # Loop until we get a regex that compiles.
        while True:
            hypothesis_messages = self.get_hypothesis_prompt(self.task_description, self.interaction_history, broken_regexes)
            hypothesis_regex_text, _ = query_api(hypothesis_messages, self.engine, self.openai_cache, self.openai_cache_file)
            hypothesis_regex_text = self.strip_hypothesis_regex(hypothesis_regex_text)
            print('Hypothesis regex (post-strip):', hypothesis_regex_text)
            try:
                hypothesis_regex = re.compile(hypothesis_regex_text)
            except re.error:
                broken_regexes.append(hypothesis_regex_text)
                print("Failed to compile hypothesis regex")
                continue
            break
        
        return hypothesis_regex

    def strip_hypothesis_regex(self, hypothesis_regex_text):
        '''Strips the hypothesis regex of quotes.
        
        Args:
            hypothesis_regex_text (str): The hypothesis regex to strip.
        
        Returns:
            str: The stripped hypothesis regex.
        '''
        hypothesis_regex_text = hypothesis_regex_text.strip('"').strip("'").strip("`")
        return hypothesis_regex_text

    @abstractmethod
    def get_hypothesis_prompt(self, interaction_history, broken_regexes=None):
        '''Creates prompt for the model which produces a hypothesis using the given active learning framework.
        
        Args:
            task_description (str): Description of the task
            interaction_history (list of tuples): List of (question, answer) tuples. The precise format of the questions / answers differs based on the type of active learning agent.
            broken_regexes (list of str): List of strings holding previous hypotheses that failed to compile.

        Returns:
            prompt (str): Prompt for the model to generate a new hypothesis
        '''
        pass
    
    @abstractmethod
    def generate_active_query(self):
        '''Generates an active query to ask the oracle.'''
        pass

    @abstractmethod
    def generate_oracle_response(self, query):
        '''Produces an oracle response to the active query, and adds (query, response) to self.interaction_history.'''
        pass
    
    def update_interaction_history(self, active_query, oracle_response):
        '''Updates self.interaction_history based on the active query and oracle response.'''
        self.interaction_history.append((active_query, oracle_response))

    def add_turn(self, query, response):
        '''Add (query, response) to self.interaction_history.'''
        self.interaction_history.append((query, response))
    
    def get_query_prompt(self):
        pass

    def get_oracle_prompt(self, question, question_type):
        answer_description = "Answer the question in the shortest way with minimal additional explanation."
        oracle_prompt = textwrap.dedent('''\
            {persona} {answer_description}
            {question}'''
        ).format(
            persona=self.persona,
            answer_description=answer_description,
            question=question
        )
        print(oracle_prompt)
        print("===")
        return oracle_prompt

    def query_oracle_api(self, question, question_type):
        oracle_prompt = self.get_oracle_prompt(question, question_type)
        answer, _ = query_api([{"role": "user", "content": oracle_prompt}], self.engine, self.openai_cache, self.openai_cache_file, temperature=self.temperature)
        return answer

    def evaluate_condition(self, **kwargs):
        return True
    
    def get_interaction_features(self):
        """
        Returns a dictionary of features for the current interaction trajectory.

        The features are:
        - interaction_time: total time spent interacting with the system (in minutes)
        - interaction_num_turns: number of turns in the interaction
        - interaction_total_char_length: total number of characters in the user's messages
        """
        return {
            "interaction_num_turns": len(self.interaction_history),
        }