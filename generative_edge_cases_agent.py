import textwrap

from base_active_learning_agent import BaseActiveLearningAgent
from utils import query_api

IMPLEMENTATION = "system"  #["Python regex", "system"]


class GenerativeEdgeCasesAgent(BaseActiveLearningAgent):
    """Active learning agent that generates edge cases to identify the target regex."""

    def __init__(self, target_specification_file, engine, openai_cache_file=None, **kwargs):
        super().__init__(target_specification_file, engine, openai_cache_file, **kwargs)

    @staticmethod
    def format_edge_cases(edge_cases):
        return '\n'.join([f"- {edge_case[0]} -> {edge_case[1]}" for edge_case in edge_cases])
    
    @staticmethod
    def strip_edge_case(edge_case):
        # Strip label
        edge_case = edge_case.split(" ->")[0]
        # Strip beginning dashes
        if edge_case.startswith("- "):
            edge_case = edge_case[2:]
        edge_case = edge_case.strip('"')
        return edge_case

    def get_hypothesis_prompt(self, task_description, interaction_history, broken_regexes=None):
        hypothesis_prompt = textwrap.dedent('''\
            Your task is to collaboratively help someone design a regex that will {task_description}.

            Help them come up with a hypothesis for the regex that they should try, consistent with the previous edge cases.

            Previous edge cases:
            {edge_cases}
            
            Previous invalid attempts (these regexes failed to compile):
            {previous_hypotheses}

            Generate the hypothesis regex without quotes and nothing else:'''
        ).format(
            task_description=task_description,
            edge_cases=self.format_edge_cases(interaction_history),
            previous_hypotheses='\n'.join(broken_regexes),
        )
        print(hypothesis_prompt)
        return [{"role": "user", "content": hypothesis_prompt}]
    
    def get_query_prompt(self):
        return self.get_edge_case_prompt(self.task_description, [["[Q]", "[A]"]], self.example_edge_case_question, self.example_edge_case_question_format)

    def get_edge_case_prompt(self, task_description, interaction_history, example_edge_case_question, example_edge_case_question_format):
        edge_case_template = textwrap.dedent('''\
            Your task is to {task_description}.
            
            Come up with a potential edge case to learn as much information as you can about what their desired behavior should be under different circumstances.
            Make sure the edge case addresses different aspects of the {implementation} than the edge cases that have already been considered.
            
            An example edge case is: {example_edge_case_question}

            Current cases:
            {edge_cases}

            Generate the most informative edge case that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Generate the edge case in the following format, and nothing else: "{example_edge_case_question_format}"'''
        ).format(
            task_description=task_description,
            implementation=IMPLEMENTATION,
            example_edge_case_question=example_edge_case_question,  # TODO have different ones
            example_edge_case_question_format=example_edge_case_question_format,
            edge_cases=self.format_edge_cases(interaction_history),
        )
        print(edge_case_template)
        print("===")
        return [{"role": "user", "content": edge_case_template}]

    def generate_active_query(self):
        '''Generates an informative edge case for the oracle.'''
        edge_case_prompt = self.get_edge_case_prompt(self.task_description, self.interaction_history, self.example_edge_case_question, self.example_edge_case_question_format)
        edge_case, _ = query_api(edge_case_prompt, self.engine, self.openai_cache, self.openai_cache_file, temperature=self.temperature)
        edge_case = self.strip_edge_case(edge_case)
        return edge_case
       
    def generate_oracle_response(self, edge_case):
        '''Generates an oracle response for the edge case.'''
        if hasattr(self, 'gold_regex'):
            edge_case = edge_case[self.example_edge_case_question_format.find("[edge case]"):].strip()
            edge_case_passes_gold = self.gold_regex.fullmatch(edge_case) is not None
        else:
            edge_case_passes_gold = super().query_oracle_api(edge_case, "samples")
        self.interaction_history.append((edge_case, edge_case_passes_gold))
        return edge_case_passes_gold
    
    def query_type(self):
        return f"edge_cases"
    
    def get_prompt(self):
        return self.get_edge_case_prompt(self.task_description, self.interaction_history, self.example_edge_case_question, self.example_edge_case_question_format)
