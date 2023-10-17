import os
import sys

# Ensure this can be run from the root directory.
sys.path.append('.')

import openai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pool_based_agent import PoolBasedAgent
from generative_questions_agent import GenerativeQuestionsAgent
from generative_edge_cases_agent import GenerativeEdgeCasesAgent
import json
import random

load_dotenv()



# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)


ENGINE =  "gpt-4"

SAVE_DIR = f"annotations_{ENGINE}"
os.makedirs(SAVE_DIR, exist_ok=True)

query_type_to_agent = {
    "Non-interactive": "non-interactive",
    "Supervised Learning": PoolBasedAgent,
    "Pool-based Active Learning": PoolBasedAgent,
    "Generative edge cases": GenerativeEdgeCasesAgent,
    "Generative open-ended questions": GenerativeQuestionsAgent,
    "Generative yes/no questions": GenerativeQuestionsAgent,
}

query_type_to_instruction = {
    "Non-interactive": "To the best of your ability, please explain all details about %task_description%, such that someone reading your responses can understand and make judgments as close to your own as possible. %noninteractive_task_description%\n<b>Note:</b> You will have up to 5 minutes to articulate your preferences. Please try to submit your response within that time. After you submit, you will be taken to the final part of the study.",
    "Supervised Learning": "Try to answer in a way that accurately and comprehensively conveys your preferences, such that someone reading your responses can understand and make judgments as close to your own as possible. Feel free to respond naturally (you can use commas, short phrases, etc), and press [enter] to send your response. Note that the chatbot technology is imperfect, and you are free to avoid answering any questions that are overly broad or uncomfortable. When interacting with the chatbot, please avoid asking follow-up questions or engaging in open-ended dialogue as the chatbot is unable to respond to you.\n<b>Note:</b> The chatbot will stop asking questions after 5 minutes, after which you can send your last response and you will be taken to the final part of the study.",
    "Pool-based Active Learning": "Try to answer in a way that accurately and comprehensively conveys your preferences, such that someone reading your responses can understand and make judgments as close to your own as possible. Feel free to respond naturally (you can use commas, short phrases, etc), and press [enter] to send your response. Note that the chatbot technology is imperfect, and you are free to avoid answering any questions that are overly broad or uncomfortable. When interacting with the chatbot, please avoid asking follow-up questions or engaging in open-ended dialogue as the chatbot is unable to respond to you.\n<b>Note:</b> The chatbot will stop asking questions after 5 minutes, after which you can send your last response and you will be taken to the final part of the study.",
    "Generative edge cases": "This chatbot will ask you a series of questions about %task_description%. Try to answer in a way that accurately and comprehensively conveys your preferences, such that someone reading your responses can understand and make judgments as close to your own as possible. Feel free to respond naturally (you can use commas, short phrases, etc), and press [enter] to send your response. Note that the chatbot technology is imperfect, and you are free to avoid answering any questions that are overly broad or uncomfortable. When interacting with the chatbot, please avoid asking follow-up questions or engaging in open-ended dialogue as the chatbot is unable to respond to you.\n<b>Note:</b> The chatbot will stop asking questions after 5 minutes, after which you can send your last response and you will be taken to the final part of the study.",
    "Generative open-ended questions": "This chatbot will ask you a series of questions about %task_description%. Try to answer in a way that accurately and comprehensively conveys your preferences, such that someone reading your responses can understand and make judgments as close to your own as possible. Feel free to respond naturally (you can use commas, short phrases, etc), and press [enter] to send your response. Note that the chatbot technology is imperfect, and you are free to avoid answering any questions that are overly broad or uncomfortable. When interacting with the chatbot, please avoid asking follow-up questions or engaging in open-ended dialogue as the chatbot is unable to respond to you.\n<b>Note:</b> The chatbot will stop asking questions after 5 minutes, after which you can send your last response and you will be taken to the final part of the study.",
    "Generative yes/no questions": "This chatbot will ask you a series of questions about %task_description%. Try to answer in a way that accurately and comprehensively conveys your preferences, such that someone reading your responses can understand and make judgments as close to your own as possible. Feel free to respond naturally (you can use commas, short phrases, etc), and press [enter] to send your response. Note that the chatbot technology is imperfect, and you are free to avoid answering any questions that are overly broad or uncomfortable. When interacting with the chatbot, please avoid asking follow-up questions or engaging in open-ended dialogue as the chatbot is unable to respond to you.\n<b>Note:</b> The chatbot will stop asking questions after 5 minutes, after which you can send your last response and you will be taken to the final part of the study.",
}


def initialize_agent_by_query_type(query_type, problem_instance_filename, pool_fp, pool_al_sampling_type, pool_diversity_num_clusters):
    question_type = "yn" if query_type == "Generative yes/no questions" else "open"
    if query_type == "Pool-based Active Learning":
        engine = "text-curie-001"
        # cache responses for efficiency when doing pool-based active learning
        openai_cache_file = f"{engine}-cache.jsonl"
    else:
        openai_cache_file = None
        engine = ENGINE
    if query_type == "Generative edge cases":
        temperature = 0.8
    else:
        temperature = 0.0
    if type(query_type_to_agent[query_type]) == str:
        return query_type_to_agent[query_type]
    if query_type == "Supervised Learning":
        pool_al_sampling_type = "random"
    return query_type_to_agent[query_type](
        problem_instance_filename,
        engine,
        openai_cache_file=openai_cache_file,
        question_type=question_type,
        pool_fp=pool_fp,
        pool_al_sampling_type=pool_al_sampling_type,
        pool_diversity_num_clusters=pool_diversity_num_clusters,
        temperature=temperature,
    )
    

experiment_type_to_prolific_id = json.load(open(f"{SAVE_DIR}/experiment_type_to_prolific_id.json"))
prompt_type_to_prompt = {}
for prompt_type in experiment_type_to_prolific_id:
    with open(f"human_exps_prompts/{prompt_type}.json") as f:
        prompt_type_to_prompt[prompt_type] = json.load(f)

def load_prolific_id_info_from_file():
    prolific_id_to_user_responses = {}
    prolific_id_to_experiment_type = {}

    for filename in os.listdir(SAVE_DIR):
        if filename.endswith(".json") and filename != "experiment_type_to_prolific_id.json":
            prolific_id = os.path.split(filename)[-1].split(".json")[0]
            with open(os.path.join(SAVE_DIR, filename)) as f:
                prolific_id_to_user_responses[prolific_id] = json.load(f)

    for prompt_type in experiment_type_to_prolific_id:
        for query_type in experiment_type_to_prolific_id[prompt_type]:
            for prolific_id in experiment_type_to_prolific_id[prompt_type][query_type]:
                prolific_id_to_experiment_type[prolific_id] = {
                    "prompt": prompt_type_to_prompt[prompt_type],
                    "query_type": query_type,
                    "agent": initialize_agent_by_query_type(
                        query_type,
                        problem_instance_filename=os.path.join("gpt_prompts/", prompt_type, random.choice(os.listdir(f"gpt_prompts/{prompt_type}"))),
                        pool_fp=(prompt_type_to_prompt[prompt_type].get("full_data_path", None) if query_type == "Supervised Learning" else prompt_type_to_prompt[prompt_type].get("pool_data_path", None)),
                        pool_al_sampling_type=("random" if query_type == "Supervised Learning" else prompt_type_to_prompt[prompt_type].get("pool_al_sampling_type", None)),
                        pool_diversity_num_clusters=prompt_type_to_prompt[prompt_type].get("pool_diversity_num_clusters", None),
                    ),
                }

    return prolific_id_to_user_responses, experiment_type_to_prolific_id, prolific_id_to_experiment_type

(
    prolific_id_to_user_responses,  # has fully completed at init
    experiment_type_to_prolific_id,  # has partially completed at init
    prolific_id_to_experiment_type,  # has partially completed at init
) = load_prolific_id_info_from_file()



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_next_prompt", methods=["POST"])
def get_next_prompt():
    prolific_id = request.form.get("prolific_id")
    error = {}
    if prolific_id in prolific_id_to_experiment_type:
        curr_prompt = prolific_id_to_experiment_type[prolific_id]["prompt"]
        curr_query_type = prolific_id_to_experiment_type[prolific_id]["query_type"]
        prolific_id_to_user_responses[prolific_id] = {
            "prolific_id": prolific_id,
            "engine": ENGINE,
            "query_type": curr_query_type,
            "prompt": curr_prompt["prompt"],
            "conversation_history": [],
            "evaluation_results": [],
            "feedback": {},
        }
        error = {"error": "This username already exists"}
    else:
        experiment_types_with_fewest_participants = []
        min_num_participants = float("inf")
        prompt_types_to_consider = ["website_preferences"]
        for prompt_type in prompt_types_to_consider:
            for query_type in experiment_type_to_prolific_id[prompt_type]:
                num_participants = len(experiment_type_to_prolific_id[prompt_type][query_type])
                if num_participants < min_num_participants:
                    experiment_types_with_fewest_participants = [(prompt_type, query_type)]
                    min_num_participants = num_participants
                elif num_participants == min_num_participants:
                    experiment_types_with_fewest_participants.append((prompt_type, query_type))

        # sample experiment to run based on the experiment type with the fewest participants
        curr_prompt_type, curr_query_type = random.choice(experiment_types_with_fewest_participants)

        curr_prompt = prompt_type_to_prompt[curr_prompt_type]
        prolific_id_to_user_responses[prolific_id] = {
            "prolific_id": prolific_id,
            "engine": ENGINE,
            "query_type": curr_query_type,
            "prompt": curr_prompt["prompt"],
            "conversation_history": [],
            "evaluation_results": [],
            "feedback": {},
        }
        prolific_id_to_experiment_type[prolific_id] = {
            "prompt": curr_prompt,
            "query_type": curr_query_type,
            "agent": initialize_agent_by_query_type(
                curr_query_type,
                problem_instance_filename=os.path.join("gpt_prompts", curr_prompt_type, random.choice(os.listdir(f"gpt_prompts/{curr_prompt_type}"))),
                pool_fp=(prompt_type_to_prompt[prompt_type].get("full_data_path", None) if query_type == "Supervised Learning" else prompt_type_to_prompt[prompt_type].get("pool_data_path", None)),
                pool_al_sampling_type=("random" if query_type == "Supervised Learning" else prompt_type_to_prompt[prompt_type].get("pool_al_sampling_type", None)),
                pool_diversity_num_clusters=curr_prompt.get("pool_diversity_num_clusters", None),
            ),
        }
        experiment_type_to_prolific_id[curr_prompt_type][curr_query_type].append(prolific_id)

        json.dump(experiment_type_to_prolific_id, open(f"{SAVE_DIR}/experiment_type_to_prolific_id.json", "w"), indent=4)
    
    prompt_to_display = [
        curr_prompt["prompt"]["preamble"],
        query_type_to_instruction[curr_query_type].replace("%task_description%", curr_prompt["prompt"]["task_description"]).replace("%noninteractive_task_description%", curr_prompt["prompt"].get("noninteractive_task_description", "")),
        curr_prompt["prompt"]["final"],
    ]
    prompt_to_display = "\n".join(prompt_to_display)
    agent = prolific_id_to_experiment_type[prolific_id]["agent"]
    if type(agent) == str:
        prolific_id_to_user_responses[prolific_id]["query_prompt"] = agent
    else:
        prolific_id_to_user_responses[prolific_id]["query_prompt"] = agent.get_query_prompt()

    return jsonify({
        "prompt": prompt_to_display,
        "evaluation_prompt": curr_prompt["prompt"]["evaluation"],
        "test_samples": curr_prompt["test_samples"],
        "mode": "prompt" if curr_query_type == "Non-interactive" else "chat",
        **error,
    })


@app.route("/update", methods=["POST"])
def update():
    """
    Sends user message (if exists) and queries active learning agent for next query
    """
    user_message = request.form.get("user_message")
    prolific_id = request.form.get("prolific_id")
    if user_message:
        if prolific_id_to_experiment_type[prolific_id]["query_type"] != "Non-interactive":
            previous_query = prolific_id_to_user_responses[prolific_id]["conversation_history"][-1]["message"]
            prolific_id_to_experiment_type[prolific_id]["agent"].add_turn(previous_query, user_message)
        assistant_display_timestamp = int(request.form.get("last_assistant_message_display_time"))
        user_submission_timestamp = int(request.form.get("last_user_message_submission_time"))
        user_time_spent_on_message = user_submission_timestamp - assistant_display_timestamp
        prolific_id_to_user_responses[prolific_id]["conversation_history"].append({
            "sender": "user",
            "message": user_message,
            "time_spent_ms": user_time_spent_on_message,
            "display_time": assistant_display_timestamp,
            "submission_time": user_submission_timestamp,
        })
    query = None
    if not request.form.get("time_up"):
        query = prolific_id_to_experiment_type[prolific_id]["agent"].generate_active_query()
        prolific_id_to_user_responses[prolific_id]["conversation_history"].append({"sender": "assistant", "message": query})

    return jsonify({"response": query})


@app.route("/update_user_response", methods=["POST"])
def update_user_response():
    user_message = request.form.get("user_message")
    prolific_id = request.form.get("prolific_id")
    previous_query = prolific_id_to_user_responses[prolific_id]["conversation_history"][-1]["message"]
    prolific_id_to_experiment_type[prolific_id]["agent"].add_turn(previous_query, user_message)
    prolific_id_to_user_responses[prolific_id]["conversation_history"].append({"sender": "user", "message": user_message})
    return jsonify({"response": "done"})


@app.route("/get_next_query", methods=["POST"])
def get_next_query():
    prolific_id = request.form.get("prolific_id")
    query = prolific_id_to_experiment_type[prolific_id]["agent"].generate_active_query()
    prolific_id_to_user_responses[prolific_id]["conversation_history"].append({"sender": "assistant", "message": query})

    return jsonify({"response": query})


@app.route("/save", methods=["POST"])
def save():
    prolific_id = request.form.get("prolific_id")
    with open(os.path.join(SAVE_DIR, f"{prolific_id}.json"), "w") as f:
        json.dump(prolific_id_to_user_responses[prolific_id], f, indent=2)
    return jsonify({"response": "done"})


@app.route("/submit_evaluation", methods=["POST"])
def evaluation_submission():
    prolific_id = request.form.get("prolific_id")
    user_labels = []
    for idx, test_sample in enumerate(prolific_id_to_experiment_type[prolific_id]["prompt"]["test_samples"]):
        user_labels.append({
            "sample": test_sample,
            "label": request.form.get(f"test-case-{idx}"),
            "explanation": request.form.get(f"test-case-{idx}-explanation"),
        })
    prolific_id_to_user_responses[prolific_id]["evaluation_results"] = user_labels
    save()
    return jsonify({"response": "done"})


@app.route("/submit_feedback", methods=["POST"])
def feedback_submission():
    prolific_id = request.form.get("prolific_id")
    for feedback_type in request.form:
        if feedback_type.startswith("feedback_"):
            prolific_id_to_user_responses[prolific_id]["feedback"][feedback_type] = request.form.get(feedback_type)
    save()
    return jsonify({"response": "done"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
