import textwrap

from base_active_learning_agent import BaseActiveLearningAgent
from utils import query_api, load_openai_cache, async_query_api
import numpy as np
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import json


IMPLEMENTATION = "system"  #["Python regex", "system"]


class PoolBasedAgent(BaseActiveLearningAgent):
    """Active learning agent that generates edge cases to identify the target regex."""

    def __init__(self, target_specification_file, engine, openai_cache_file=None, pool_data_path=None, pool_al_sampling_type=None, pool_diversity_num_clusters=None, **kwargs):
        super().__init__(target_specification_file, engine, openai_cache_file, **kwargs)
        # either specified in `target_specification_file` or in args
        if pool_data_path is not None:
            self.pool_data_path = pool_data_path
        if pool_al_sampling_type is not None:
            self.pool_al_sampling_type = pool_al_sampling_type
        self.pool_al_examples = self.load_pool_examples(self.pool_data_path)
        self.previous_samples = []
        if self.pool_al_sampling_type == "diversity":
            self.num_clusters = pool_diversity_num_clusters
            print("Loading sentence transformer model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding pool examples...")
            # embed everything
            self.pool_al_examples_embeddings = model.encode(self.pool_al_examples)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(self.pool_al_examples_embeddings)
            # get centroids of clusters
            centroids = kmeans.cluster_centers_
            # get closest example to each centroid
            self.all_samples = []
            # round robin
            self.curr_centroid_idx = 0
            for centroid_idx, centroid in enumerate(centroids):
                # closest_example_idx = np.argmin(np.linalg.norm(self.pool_al_examples_embeddings - centroid, axis=1))
                cluster_samples = np.where(kmeans.labels_ == centroid_idx)[0]
                # sort by distance (smallest to largest)
                cluster_samples = cluster_samples[np.argsort(np.linalg.norm(self.pool_al_examples_embeddings[cluster_samples] - centroid, axis=1))]
                self.all_samples.append([self.pool_al_examples[pool_sample] for pool_sample in cluster_samples])
            all_samples = []
            for sample in self.all_samples: all_samples.extend(sample)
            assert set(all_samples) == set(self.pool_al_examples)
        if self.pool_al_sampling_type == "uncertainty_logits":
            self.engine_selection = "text-davinci-003"
            self.openai_cache_selection_file = f"{self.engine_selection}-cache.jsonl"
            self.openai_cache_selection = load_openai_cache(self.openai_cache_selection_file)

    
    def load_pool_examples(self, pool_fp):
        # csv_reader = csv.DictReader(open(pool_fp, 'r'), delimiter='\t')
        pool_examples = []
        for row in open(pool_fp):
            pool_examples.append(json.loads(row)["nl_desc"])
        return pool_examples

    def format_edge_cases(self, edge_cases):
        return '\n'.join([f"{idx+1}. {edge_case[0]} -> {edge_case[1]}" for idx, edge_case in enumerate(edge_cases)])
    
    def format_al_json_samples(self, edge_cases):
        return json.dumps([{"sample": sample.strip()} for sample in edge_cases])
    
    @staticmethod
    def strip_edge_case(edge_case):
        # Strip label
        edge_case = edge_case.split(" -> ")[0]
        # Strip beginning dashes
        if edge_case.startswith("- "):
            edge_case = edge_case[2:]
        return edge_case
    
    def get_hypothesis_prompt(self):
        pass
    
    def get_query_prompt(self):
        return f"pool_{self.pool_al_sampling_type}"

    def generate_active_query(self):
        '''Generates the next active learning query.'''
        if self.pool_al_sampling_type == "uncertainty_tokens":
            sample = self.generate_active_query_uncertainty_tokens(batch_size=10)
        elif self.pool_al_sampling_type == "uncertainty_logits":
            sample = self.generate_active_query_uncertainty_logits()
        elif self.pool_al_sampling_type == "diversity":
            sample = self.generate_active_query_diversity()
        elif self.pool_al_sampling_type == "random":
            sample = self.generate_active_query_random()
        else:
            raise NotImplementedError
        self.previous_samples.append(sample)
        self.pool_al_examples.remove(sample)
        print(sample)
        print("===")
        return self.example_edge_case_question_format.replace("[edge case]", sample)
        
    def generate_active_query_diversity(self):
        # make people go through a fixed number (k turns)
        # if len(self.previous_samples) >= len(self.all_samples):
        #     return self.generate_active_query_random()
        next_sample = self.all_samples[self.curr_centroid_idx].pop(0)
        self.curr_centroid_idx = (self.curr_centroid_idx + 1) % self.num_clusters
        return next_sample
    
    def generate_active_query_random(self):
        random_sample = random.choice(self.pool_al_examples)
        return random_sample

    def generate_active_query_uncertainty_tokens(self, batch_size):
        '''Samples the most uncertain edge case for the oracle.'''
        """
        TODO old code... remove
        """
        most_uncertain_edge_case = None
        max_uncertainty = 0
        for possible_next_edge_case_idx in tqdm(range(0, len(self.pool_al_examples), batch_size)):
            next_edge_cases = self.pool_al_examples[possible_next_edge_case_idx:possible_next_edge_case_idx+batch_size]
            al_template = textwrap.dedent('''\
                {pool_al_prompt}
                {previous_examples}
                
                {pool_al_prompt2}
                {next_edge_cases}

                Return a json list of the form [{{"sample": sample, "pred label": yes/no, "pred prob": probability of predicted label for the sample}}]. Please stick to this format and return nothing else.'''
            ).format(
                pool_al_prompt=self.pool_al_prompt[0],
                previous_examples=self.format_edge_cases([
                    [self.previous_samples[idx], item[1]] for idx, item in enumerate(self.interaction_history)
                ]),
                pool_al_prompt2=self.pool_al_prompt[1],
                next_edge_cases=self.format_al_json_samples(next_edge_cases),
            )
            response, _ = query_api(
                [{"role": "user", "content": al_template}],
                # "gpt-3.5-turbo",
                self.engine,
                openai_cache=self.openai_cache,
                openai_cache_file=self.openai_cache_file,
                max_tokens=1000,
            )
            try:
                responses = json.loads(response)
            except:
                breakpoint()
            for sample in responses:
                prob_positive = sample["pred prob"]
                uncertainty = -prob_positive * np.log(prob_positive) - (1 - prob_positive) * np.log(1 - prob_positive)
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                    most_uncertain_edge_case = sample["sample"]
        return most_uncertain_edge_case


    def generate_active_query_uncertainty_logits(self):
        '''Samples the most uncertain edge case for the oracle.'''
        all_uncertainty_prompts_positive = []
        all_uncertainty_prompts_negative = []
        for possible_next_edge_case in self.pool_al_examples:
            al_template = textwrap.dedent('''\
                {pool_al_prompt}
                {previous_examples}

                {pool_al_prompt2}
                {current_example}'''
            ).format(
                pool_al_prompt=self.pool_al_prompt[0],
                previous_examples=self.format_edge_cases([
                    [self.previous_samples[idx], item[1]] for idx, item in enumerate(self.interaction_history)
                ]),
                pool_al_prompt2=self.pool_al_prompt[1],
                current_example=self.format_edge_cases([[possible_next_edge_case, ""]]),
            )
            all_uncertainty_prompts_positive.append([al_template + "yes"])
            all_uncertainty_prompts_negative.append([al_template + "no"])

        edge_case_yes_to_response_text, edge_case_yes_to_response = async_query_api(
            all_uncertainty_prompts_positive,
            self.engine_selection,
            openai_cache=self.openai_cache_selection,
            openai_cache_file=self.openai_cache_selection_file,
            max_tokens=0,
            echo=True,
            logprobs=0,
        )
        _, edge_case_no_to_response = async_query_api(
            all_uncertainty_prompts_negative,
            self.engine_selection,
            openai_cache=self.openai_cache_selection,
            openai_cache_file=self.openai_cache_selection_file,
            max_tokens=0,
            echo=True,
            logprobs=0,
        )

        most_uncertain_edge_case = None
        max_uncertainty = 0
        for possible_next_edge_case, prompt_yes, prompt_no in zip(self.pool_al_examples, all_uncertainty_prompts_positive, all_uncertainty_prompts_negative):
            # for label in ["Yes", "No"]:
            yes_responses = edge_case_yes_to_response[json.dumps(prompt_yes)]
            assert yes_responses['choices'][0]['logprobs']['tokens'][-1].strip() == "yes"
            yes_logprob = yes_responses['choices'][0]['logprobs']['token_logprobs'][-1]
            no_responses = edge_case_no_to_response[json.dumps(prompt_no)]
            assert no_responses['choices'][0]['logprobs']['tokens'][-1].strip() == "no"
            no_logprob = no_responses['choices'][0]['logprobs']['token_logprobs'][-1]

            all_logprobs = [yes_logprob, no_logprob]
            probs = np.exp(np.array(all_logprobs))
            prob_positive = probs[0] / probs.sum()
            uncertainty = -prob_positive * np.log(prob_positive) - (1 - prob_positive) * np.log(1 - prob_positive)
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                most_uncertain_edge_case = possible_next_edge_case
        return most_uncertain_edge_case
       
    def generate_oracle_response(self, edge_case):
        '''Generates an oracle response for the edge case.'''
        if hasattr(self, 'gold_regex'):
            breakpoint()
            edge_case = edge_case.strip()
            edge_case_passes_gold = self.gold_regex.fullmatch(edge_case) is not None
        else:
            edge_case_passes_gold = super().query_oracle_api(edge_case, "samples")
        self.interaction_history.append((edge_case, edge_case_passes_gold))
        return edge_case_passes_gold
