import glob
import sys

import numpy as np
from matplotlib import pyplot as plt
from tap import Tap

from from_saved_file_agent import FromSavedFileAgent
from main import run_problem_instance
import json
import os
from tqdm import tqdm
from scipy import stats
import random
import wandb
from sklearn.metrics import auc
from utils import average_lines
from scipy.interpolate import interp1d

import pandas as pd
import seaborn as sns

log_wandb = False

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Get the default color cycle

DO_PLOT = True
AGENT_NAME_TO_CLASS = {
    "saved_file": FromSavedFileAgent,
}
BAR_PLOT_METRIC = "correct_prob_relative"

task_specific_directives = {
    "website_preferences": '\nFor this task, "yes" means the user would like the website, and "no" means the user would not like the website',
    "moral_reasoning": '\nFor this task, "yes" means the user would believe it is ethical to steal a loaf of bread, and "no" means the user would believe it is not ethical to steal a loaf of bread',
    "email_regex": '\nFor this task, "yes" means the user would find the email address valid, while "no" means the user would find the email address invalid',
}
task_specific_instructions = {
    "website_preferences": "asks a user about their preferences for a website",
    "moral_reasoning": "asks a user under what conditions they would believe it is ethical to steal a loaf of bread",
    "email_regex": "asks a user about their preferences for what makes a valid format for email addresses",
}

COLORS = {
    "Generative edge cases": "tab:blue",
    "Generative yes/no questions": "tab:orange",
    "Generative open-ended questions": "tab:green",
    "Pool-based Active Learning": "#767676",
    "Non-interactive": "#a5a5a5",
}


def get_saved_interaction_files_for_task(task):
    saved_annotations_dir = "annotations_gpt-4"
    with open(f"{saved_annotations_dir}/experiment_type_to_prolific_id.json") as f:
        experiment_type_to_prolific_id = json.load(f)
    files_to_return = {}
    for experiment_type in experiment_type_to_prolific_id[task]:
        files_to_return[experiment_type] = [
            f"{saved_annotations_dir}/{file_id}.json" for file_id in experiment_type_to_prolific_id[task][experiment_type] if os.path.exists(f"{saved_annotations_dir}/{file_id}.json")
        ]
    return files_to_return


def main(args):
    wandb.login()
    if args.no_cache:
        openai_cache_file = None
    else:
        openai_cache_file = f"{args.engine}-cache-seed-{args.seed}.jsonl"

    all_test_xs = {
        "interaction_time": {},
        "interaction_num_turns": {},
        "interaction_total_char_length": {},
    }
    all_test_scores = {
        "accuracy": {},
        "AUCROC": {},
        "correct_prob": {},
        "accuracy_relative": {},
        "AUCROC_relative": {},
        "correct_prob_relative": {}
    }

    #  initialize a dataframe
    all_test_results = pd.DataFrame(columns=[
        'interaction_time', 'interaction_num_turns', 'interaction_total_char_length',
        'accuracy', 'AUCROC', 'correct_prob', 'accuracy_relative', 'AUCROC_relative',
        'correct_prob_relative', 'question_mode', 'task', 'engine', 'seed', 'interaction_id',
    ])

    problem_instance_filename = random.choice(glob.glob(f"gpt_prompts/{args.task}/*.json"))
    saved_interaction_files_for_task = get_saved_interaction_files_for_task(args.task)
    for question_mode in saved_interaction_files_for_task:
        print(question_mode)
        for metric in all_test_xs:
            all_test_xs[metric][question_mode] = []
        for metric in all_test_scores:
            all_test_scores[metric][question_mode] = []
        for saved_interactions_file in tqdm(saved_interaction_files_for_task[question_mode]):
            print(saved_interactions_file)
            # filter out preferences that are trivial
            if args.filter_trivial_preferences:
                with open(saved_interactions_file) as f:
                    saved_interactions = json.load(f)
                    all_answers = [sample["label"] for sample in saved_interactions["evaluation_results"]]
                    if len(set(all_answers)) == 1:
                        continue
            os.makedirs(f"model_human_results/{args.task}", exist_ok=True)
            outputs_save_file = open(f"model_human_results/{args.task}/{args.engine}_{args.eval_condition}_{question_mode.replace('/', '_').replace(' ', '_')}_{os.path.split(saved_interactions_file)[-1][:-5]}.txt", "w")
            test_xs, test_scores = run_problem_instance(
                problem_instance_filename=problem_instance_filename, 
                engine=args.engine, 
                openai_cache_file=openai_cache_file,
                num_interactions=sys.maxsize,
                agent_class=FromSavedFileAgent,
                question_type=None,
                sampling_type=None,
                saved_interactions_file=saved_interactions_file,
                outputs_save_file=outputs_save_file,
                base_query_type=question_mode,
                task=args.task,
                eval_condition=args.eval_condition,
            )
            outputs_save_file.write(f"===AVG TEST XS===\n{json.dumps(test_xs, indent=2)}\n\n")
            outputs_save_file.write(f"===AVG TEST SCORES===\n{json.dumps(test_scores, indent=2)}\n\n")
            for metric in test_xs:
                all_test_xs[metric][question_mode].append(test_xs[metric])
            for metric in test_scores:
                all_test_scores[metric][question_mode].append(test_scores[metric])
            all_test_results.loc[len(all_test_results)] = {
                **test_xs, **test_scores,
                'question_mode': question_mode, 'task': args.task,
                'engine': args.engine, 'seed': args.seed,
                'interaction_id': os.path.split(saved_interactions_file)[-1][:-5],
            }
            if (all_test_results["interaction_id"] != all_test_results["interaction_id"]).any():
                breakpoint()

    for question_mode in all_test_scores["AUCROC"]:
        print(question_mode)
        print("AUCROC: " + str(all_test_scores["AUCROC_relative"][question_mode]))
        print("Accuracy: " + str(all_test_scores["accuracy_relative"][question_mode]))
        print("Prob of correct answer: " + str(all_test_scores["correct_prob_relative"][question_mode]))

    # save to file
    os.makedirs(f"model_human_results/{args.task}", exist_ok=True)
    all_test_results.to_csv(f"model_human_results/{args.task}/{args.engine}_{args.eval_condition}_all_test_results.csv")



class ArgumentParser(Tap):
    # num_interactions: int = 20  # The number of interactions between the AL agent and the oracle.
    task: str = "moral_reasoning"  # The target format we are designing for experiments (e.g. email_regex, moral_reasoning, website_preferences)
    eval_condition: str = "per_minute"  # When to evaluate the agent (e.g. at_end, per_minute, per_turn, per_turn_up_to_5)
    engine: str = "gpt-4"  # The OpenAI engine to use (e.g. gpt-3.5-turbo, gpt-4).
    no_cache: bool = False  # Whether to use the OpenAI cache file.
    batch_eval: bool = False  # Whether to batch eval (faster but less accurate)
    seed: int = 0  # The random seed to use.
    filter_trivial_preferences: bool = False  # Whether to filter out trivial preferences (e.g. all yes or all no)


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)