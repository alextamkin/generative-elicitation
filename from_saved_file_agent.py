
import textwrap
import json

from base_active_learning_agent import BaseActiveLearningAgent
from utils import query_api, load_openai_cache

IMPLEMENTATION = "system"  #["Python regex", "system"]

class FromSavedFileAgent(BaseActiveLearningAgent):
    """Agent that loads generated interactions (queries and answers) from a saved file."""

    def __init__(self, target_specification_file, engine, openai_cache_file=None, saved_interactions_file=None, eval_condition="at_end", base_query_type="questions", **kwargs):
        super().__init__(target_specification_file, engine, openai_cache_file)

        self.saved_interactions_file = saved_interactions_file
        self.saved_interactions_contents = json.load(open(saved_interactions_file))
        self.interaction_history = []
        self.initialize_test_cases(self.saved_interactions_contents["evaluation_results"])
        self.noninteractive_mode = self.saved_interactions_contents.get("query_type") == "Non-interactive"
        self.initialize_full_interaction_history(self.saved_interactions_contents["conversation_history"])
        self.eval_condition = eval_condition
        self.last_eval_turn = 0
        self.base_query_type = base_query_type
        self.query_prompt = self.saved_interactions_contents.get("query_prompt")
 
    def initialize_test_cases(self, human_test_cases):
        self.test_cases = [
            (test_case["sample"], test_case["label"] == "yes")
            for test_case in human_test_cases
        ]

    def initialize_full_interaction_history(self, human_interactions):
        self.turn_timings = {"user": [], "assistant": []}
        if self.noninteractive_mode:
            assert len(human_interactions) == 1
            self.full_interaction_history = [("", turn["message"]) for turn in human_interactions]
            self.turn_timings["user"].append(human_interactions[0].get("time_spent_ms", None))
        else:
            self.full_interaction_history = []
            for turn in human_interactions:
                if turn["sender"] == "assistant":
                    self.full_interaction_history.append((turn["message"], None))
                else:
                    self.full_interaction_history[-1] = (self.full_interaction_history[-1][0], turn["message"])
                self.turn_timings[turn["sender"]].append(turn.get("time_spent_ms", None))
            # Remove the last turn if it's empty
            if self.full_interaction_history[-1][1] is None:
                self.full_interaction_history.pop()
                self.turn_timings["assistant"].pop()

    def format_questions_and_answers(self, questions_and_answers):
        '''Formats the questions and answers into a string.

        Looks like:
        - Should the system allow numbers in the domain? -> Yes

        Args:
            questions_and_answers (list): A list of tuples of the form (question, answer).
        
        Returns:
            str: The formatted questions and answers.
        '''
        if self.noninteractive_mode:
            return '\n'.join([f"- {answer}" for question, answer in questions_and_answers])
        else:
            return '\n'.join([f"- {question} -> {answer}" for question, answer in questions_and_answers])

    def generate_active_query(self):
        '''Generates a question for the oracle.'''
        if len(self.interaction_history) >= len(self.full_interaction_history):
            return None
        return self.full_interaction_history[len(self.interaction_history)][0]
       
    def generate_oracle_response(self, question):
        '''Generates an oracle response for the question'''
        if len(self.interaction_history) < len(self.full_interaction_history):
            assert question == self.full_interaction_history[len(self.interaction_history)][0]
            answer = self.full_interaction_history[len(self.interaction_history)][1]
            self.interaction_history.append((question, answer))
            return answer
        else:
            return None
    
    def get_hypothesis_prompt(self, interaction_history, broken_regexes=None):
        pass
    
    def score_test_cases(self, score_type="no_hypothesis", **kwargs):
        self.last_eval_turn = len(self.interaction_history)
        return super().score_test_cases(score_type, **kwargs)

    def get_curr_user_timings_ms(self):
        return sum(self.turn_timings["user"][:len(self.interaction_history)])

    def get_curr_user_message_lengths(self):
        return sum([len(turn[1]) for turn in self.interaction_history])

    def get_interaction_features(self):
        """
        Returns a dictionary of features for the current interaction trajectory.

        The features are:
        - interaction_time: total time spent interacting with the system (in minutes)
        - interaction_num_turns: number of turns in the interaction
        - interaction_total_char_length: total number of characters in the user's messages
        """
        return {
            "interaction_time": self.get_curr_user_timings_ms() / 60 / 1000,
            "interaction_num_turns": len(self.interaction_history),
            "interaction_total_char_length": self.get_curr_user_message_lengths(),
        }

    def evaluate_condition(self, **kwargs):
        if self.eval_condition == "at_end":
            return len(self.interaction_history) == len(self.full_interaction_history)
        elif self.eval_condition == "per_minute":
            total_interaction_time_ms_curr_turn = sum(self.turn_timings["user"][:len(self.interaction_history)])
            if len(self.interaction_history) == 0:
                total_interaction_time_ms_prev_turn = 0
            else:
                total_interaction_time_ms_prev_turn = sum(self.turn_timings["user"][:self.last_eval_turn])
            return (len(self.interaction_history) == len(self.full_interaction_history)) or (total_interaction_time_ms_curr_turn // 60000 > total_interaction_time_ms_prev_turn // 60000)
        elif self.eval_condition == "per_turn_up_to_5":
            return len(self.interaction_history) <= 5
        else:
            return True

    def get_query_prompt(self):
        return self.query_prompt
