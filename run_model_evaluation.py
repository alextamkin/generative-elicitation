from copy import deepcopy
import glob

import numpy as np
import openai
from matplotlib import pyplot as plt
from tap import Tap

from generative_questions_agent import GenerativeQuestionsAgent
from generative_edge_cases_agent import GenerativeEdgeCasesAgent
from from_saved_file_agent import FromSavedFileAgent
from pool_based_agent import PoolBasedAgent
import os
import json
from tqdm import tqdm
from utils import update_metrics, update_test_responses
import wandb
log_wandb = False

import pandas as pd
import seaborn as sns


AGENT_NAME_TO_CLASS = {
    "questions": GenerativeQuestionsAgent,
    "edge": GenerativeEdgeCasesAgent,
    "saved": FromSavedFileAgent,
    "pool": PoolBasedAgent,
    "noninteractive": None,
}
AGENT_CLASS_TO_NAME = {v: k for k, v in AGENT_NAME_TO_CLASS.items()}
FILE_QUERY_TYPE_TO_NAME = {
    "Pool-based Active Learning": "pool_diversity",
    "Supervised Learning": "supervised",
    "Generative edge cases": "edge_cases",
    "Generative yes/no questions": "questions_yn",
    "Generative open-ended questions": "questions_open",
    "Non-interactive": "noninteractive",
}

COLORS = {
    "Generative edge cases": "tab:blue",
    "Generative yes/no questions": "tab:orange",
    "Generative open-ended questions": "tab:green",
    "Pool-based Active Learning": "#767676",
    "Non-interactive": "#a5a5a5",
}
# def wandb_init():


def run_problem_instance(
    problem_instance_filename, engine, openai_cache_file,
    num_interactions, agent_class, question_type, sampling_type,
    saved_interactions_file, temperature=0.0, outputs_save_file=None, base_query_type=None,
    pool_diversity_num_clusters=15, task=None, eval_condition="per_minute",
):
    '''Runs the generative active learning loop for a single problem instance.
    
    This entails doing several rounds of interaction between the AL agent and the oracle, 
    evaluating the accuracies on held-out test cases after each interaction.

    Args:
        problem_instance_filename (str): The path to the problem instance file (holds the gold regex and test cases)
        engine (str): The OpenAI engine to use (e.g. gpt-3.5-turbo, gpt-4).
        openai_cache_file (str): The path to the OpenAI cache file.
        num_interactions (int): The number of interactions between the AL agent and the oracle.
    
    Returns:
        list: The test scores after each interaction.
    '''

    generative_al_agent = agent_class(
        problem_instance_filename, engine, openai_cache_file=openai_cache_file,
        question_type=question_type, saved_interactions_file=saved_interactions_file,
        eval_condition=eval_condition, pool_al_sampling_type=sampling_type,
        pool_diversity_num_clusters=pool_diversity_num_clusters,
        # temperature=0.8 if agent_class == GenerativeEdgeCasesAgent else 0.0,
        temperature=temperature, base_query_type=base_query_type,
    )
    
    if agent_class != FromSavedFileAgent:
        outputs_save_file.write(f"0. {generative_al_agent.persona}\n\n")
        query_type = AGENT_CLASS_TO_NAME[agent_class]
        if query_type == "questions":
            query_type += "_" + question_type
        elif query_type == "pool":
            query_type += "_" + sampling_type
        persona = generative_al_agent.persona
        oracle_prompt = generative_al_agent.get_oracle_prompt("[question]", question_type)
    else:
        query_type = FILE_QUERY_TYPE_TO_NAME[base_query_type]
        persona = saved_interactions_file.split("/")[-1].split(".")[0]
        oracle_prompt = "human"
    if log_wandb:
        wandb.init(
            project="generative_al",
            config={
                "domain": task,
                "query_type": query_type,
                "persona": persona,
                "is_human": agent_class == FromSavedFileAgent,
                "engine": engine,
                "seed": openai_cache_file.split("-")[-1].split(".")[0],
                "query_prompt": generative_al_agent.get_query_prompt(),
                "oracle_prompt": oracle_prompt,
                "test_case_prompt": generative_al_agent.get_test_case_prompt(
                    [("[question]", "[answer]") for _ in range(3)],
                    ("[test_case_question]", "[test_case_answer]"),
                ),
            }
        )

    test_xs = generative_al_agent.get_interaction_features()
    test_score, test_responses = generative_al_agent.score_test_cases()
    print(test_score)
    all_test_xs = update_metrics({}, test_xs)
    test_scores = update_metrics({}, test_score)
    start_test_scores = deepcopy(test_scores)
    all_test_responses = update_test_responses([], test_responses)

    if log_wandb:
        wandb.log(test_score)
        wandb.log(test_xs)

    for i in tqdm(range(num_interactions)):
        query = generative_al_agent.generate_active_query()
        if query is None:
            break

        answer = generative_al_agent.generate_oracle_response(query)

        outputs_save_file.write(f"{i}. {query}\n{answer}\n\n")
        if not generative_al_agent.evaluate_condition():
            continue
        
        outputs_save_file.write("EVAL POINT\n")

        # hypothesis_regex = generative_al_agent.generate_hypothesis()
        test_xs = generative_al_agent.get_interaction_features()
        test_score, test_responses = generative_al_agent.score_test_cases(start_metrics=start_test_scores)
        print(test_score)
        all_test_xs = update_metrics(all_test_xs, test_xs)
        test_scores = update_metrics(test_scores, test_score)
        all_test_responses = update_test_responses(all_test_responses, test_responses)

        # for k, v in test_scores.items():
        #     wandb.log(f"test_score_{k}", v)
        # for k, v in test_xs.items():
        #     wandb.log(f"test_feats_{k}", v)
        # wandb.log(f"test_score", test_score)
        # wandb.log(f"test_feats", test_xs)
        if log_wandb:
            wandb.log(test_score)
            wandb.log(test_xs)
            wandb.log({"test_responses": test_responses})
            wandb.log({"interaction_history": generative_al_agent.interaction_history})
    
    print(test_xs)
    print(test_scores)
    outputs_save_file.write(f"===TEST RESPONSES===\n{json.dumps(all_test_responses, indent=2)}\n\n")
    if log_wandb:
        wandb.finish()
    
    return all_test_xs, test_scores


def main(args):
    wandb.login()
    if args.no_cache:
        openai_cache_file = None
    else:
        openai_cache_file = f"{args.engine}-cache-seed-{args.seed}.jsonl"

    all_test_results = pd.DataFrame(columns=[
        'interaction_time', 'interaction_num_turns', 'interaction_total_char_length',
        'accuracy', 'AUCROC', 'correct_prob', 'accuracy_relative', 'AUCROC_relative',
        'correct_prob_relative', 'question_mode', 'task', 'engine', 'seed',
    ])

    # setting = args.agent
    # if args.agent == "questions":
    #     setting += "_" + args.question_type
    # elif args.agent == "pool":
    #     setting += "_" + args.sampling_type
    if args.task == "website_preferences":
        question_modes = ["questions_open", "questions_yn", "edge_cases", "pool_diversity", "pool_random", "pool_uncertainty_logits"]
    else:
        question_modes = ["questions_open", "questions_yn", "edge_cases"]
    

    avg_test_scores = {}
    for question_mode in question_modes:
        avg_test_scores[question_mode] = {}
        print(question_mode)

        for problem_instance_filename in tqdm(glob.glob(f"gpt_prompts/{args.task}/*.json")):

            agent_class = AGENT_NAME_TO_CLASS[question_mode.split("_")[0]]
            sampling_type = None
            if question_mode.split("_")[0] == "pool":
                sampling_type = "_".join(question_mode.split("_")[1:])
            elif question_mode.split("_")[0] == "questions":
                question_type = "_".join(question_mode.split("_")[1:])

            os.makedirs(f"model_model_results/{args.task}", exist_ok=True)
            outputs_save_file = open(f"model_model_results/{args.task}/{args.engine}_{args.eval_condition}_{args.seed}_{question_mode}.txt", "w")

            test_xs, test_scores = run_problem_instance(
                problem_instance_filename=problem_instance_filename, 
                engine=args.engine, 
                openai_cache_file=openai_cache_file, 
                num_interactions=args.num_interactions, 
                agent_class=agent_class,
                temperature=args.temperature,
                question_type=question_type,
                sampling_type=sampling_type,
                pool_diversity_num_clusters=args.pool_diversity_num_clusters,
                saved_interactions_file=None,
                outputs_save_file=outputs_save_file,
                task=args.task,
            )
            avg_test_scores[question_mode] = update_metrics(avg_test_scores[question_mode], test_scores)

            all_test_results.loc[len(all_test_results)] = {
                **test_xs, **test_scores, 'question_mode': question_mode,
                'task': args.task, 'engine': args.engine, 'seed': args.seed,
            }

            outputs_save_file.write(f"===AVG TEST XS===\n{json.dumps(test_xs, indent=2)}\n\n")
            outputs_save_file.write(f"===AVG TEST SCORES===\n{json.dumps(test_scores, indent=2)}\n\n")
    
        # for question_mode in avg_test_scores:
        #     for metric in avg_test_scores[question_mode]:
        #         avg_test_scores[question_mode][metric] = np.mean(avg_test_scores[question_mode][metric], axis=0)
        # try:
        #     outputs_save_file.write(f"===AVG TEST SCORES===\n{json.dumps(avg_test_scores, indent=2)}\n\n")
        # except:
        #     breakpoint()
        # outputs_save_file.close()
        for metric in avg_test_scores[question_mode]:
            print(f'All test {metric}s:', avg_test_scores[question_mode][metric])
            avg_test_scores[question_mode][metric] = np.mean(avg_test_scores[question_mode][metric], axis=0)
            print(f'Test {metric}s across training:', np.array(avg_test_scores[question_mode][metric]))
            print(f'Avg. test {metric}s across training:', avg_test_scores[question_mode][metric])
    
    os.makedirs(f"model_model_results/{args.task}", exist_ok=True)
    all_test_results.to_csv(f"model_model_results/{args.task}/{args.engine}_{args.eval_condition}_{args.seed}_all_test_results.csv")


class ArgumentParser(Tap):
    num_interactions: int = 5  # The number of interactions between the AL agent and the oracle.
    engine: str = "gpt-4"  # The OpenAI engine to use (e.g. gpt-3.5-turbo, gpt-4).
    agent: str = "edge_cases"  # The active learning agent to use (e.g. questions, edge_cases, pool, saved_file).
    eval_condition: str = "per_turn"
    # question_type: str = "open"  # The type of question to ask -- for questions agent only (e.g. open, yn).
    # sampling_type: str = "diversity"  # The type of sampling to use -- for pool agent only (e.g. diversity, random, uncertainty_logits, uncertainty_tokens).
    pool_diversity_num_clusters: int = 5  # The number of clusters to use for diversity sampling.
    task: str = "email_regex"  # The target format we are designing for experiments (e.g. email_regex, moral_reasoning, website_preferences).
    no_cache: bool = False  # Whether to use the OpenAI cache file.
    seed: int = 0  # The random seed to use.
    temperature: float = 0.0  # The temperature to use for the model.


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)