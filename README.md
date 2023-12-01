# Eliciting Human Preferences with Language Models
Paper link: https://arxiv.org/abs/2310.11589

Abstract: Language models (LMs) can be directed to perform target tasks by using labeled examples or natural language prompts. But selecting examples or writing prompts for can be challenging--especially in tasks that involve unusual edge cases, demand precise articulation of nebulous preferences, or require an accurate mental model of LM behavior. We propose to use *LMs themselves* to guide the task specification process. In this paper, we introduce **Generative Active Task Elicitation (GATE)**: a learning framework in which models elicit and infer intended behavior through free-form, language-based interaction with users. We study GATE in three domains: email validation, content recommendation, and moral reasoning. In preregistered experiments, we show that LMs prompted to perform GATE (e.g., by generating open-ended questions or synthesizing informative edge cases) elicit responses that are often more informative than user-written prompts or labels. Users report that interactive task elicitation requires less effort than prompting or example labeling and surfaces novel considerations not initially anticipated by users. Our findings suggest that LM-driven elicitation can be a powerful tool for aligning models to complex human preferences and values.


## Setting up your environment
Run the following to set up your conda environment for running GATE:
```bash
conda create -n gate PYTHON=3.10
conda activate gate
pip install -r requirements.txt
```

## Running the elicitation interface
First, ensure that the configuration file `annotations_gpt-4/experiment_type_to_prolific_id.json` is initialized as a nested dictionary with domain names (among `website_preferences`, `moral_reasoning`, `email_regex`) as keys for the outer dictionary and elicitation methods (among `Supervised Learning`, `Pool-based Active Learning`, `Non-interactive` aka prompting, `Generative edge cases`, `Generative yes/no questions`, `Generative open-ended questions`) as keys for the inner dictionary.
```JSON
{
    "moral_reasoning": {  // domain name
        "Generative open-ended questions": [  // elicitation method
            // experiment IDs for each (domain, elicitation method) pair will populate in here
        ],
        ...
    },
    ...
}
```
A sample configuration file can be found in [`annotations_gpt-4/experiment_type_to_prolific_id.json`](https://github.com/alextamkin/generative-elicitation/tree/main/annotations_gpt-4/experiment_type_to_prolific_id.json).
Note that supervised learning and pool-based active learning require access to an existing pool, which is only available for `website_preferences` at the moment.


To launch the user interface that elicits humans for their preferences, run:
```bash
# remember to set your OpenAI API key!
export OPENAI_API_KEY = <insert-your-API-key-here>
# run the server
python WebInterface/server/webserver.py
```
The interface will randomly chooses a domain and elicitation method from among those specified in file `annotations_gpt-4/experiment_to_prolific_ids.json` and query the user for their preferences in that domain, using the elicitation method. The resulting transcript is saved under `annotations_gpt-4/`.

We are not releasing the human-model transcripts we collected at this time due to privacy concerns. However, a sample transcript can be found in the [`annotations_gpt-4/`](https://github.com/alextamkin/generative-elicitation/tree/main/annotations_gpt-4) folder in this repository. You may also create your own transcripts using the above interface.


## Evaluating elicitation methods

Given a set of elicitation transcripts and gold human responses (produced by running the elicitation interface above), we can evaluate how well a model is able to make decisions by running the command:

```bash
# remember to set your OpenAI API key!
export OPENAI_API_KEY = <insert-your-API-key-here>
# run evaluation
python run_human_evaluation.py \
    --saved_annotations_dir <saved_annotations_dir> \
    --task [website_preferences|moral_reasoning|email_regex] \
    --eval_condition [per_turn|per_minute|at_end] \
    --engine <engine>
```
where:
* `--saved_annotations_dir` points to the directory where the human transcripts are saved (e.g. `annotations_gpt-4/`).
* `--task` refers to which domain we are evaluating (content recommendation, moral reasoning, email validation).
* `--eval_condition` refers to how often we produce evaluate the intermediate results of each transcript, with `per_turn` meaning we evaluate the transcript after each turn, `per_minute` meaning we evaluate the transcript only after each minute of interaction, and `at_end` meaning we only evaluate the transcript at the very end.
* `--engine` refers to which GPT model we're using (e.g. `gpt-4`).

This prompts a language model to make decisions based on the contents of the transcript and compares them to the human-provided decisions on those same examples.


### Using LMs to simulate humans
Instead of querying real humans, we can also use a LM to *simulate* human preferences. To do so, we prompt GPT4 with a set of persona prompts (which can be found in `gpt_prompts/`). You can run the elicitation loop with simulated humans by running the command:

```bash
# remember to set your OpenAI API key!
export OPENAI_API_KEY = <insert-your-API-key-here>
# run evaluation
python run_model_evaluation.py \
    --engine <engine> \
    --agent [questions|edge_cases|pool] \
    --eval_condition [per_turn|per_minute|at_end] \
    --pool_diversity_num_clusters <pool_diversity_num_clusters> \
    --task [website_preferences|moral_reasoning|email_regex] \
```

where:
* `--engine` refers to which GPT model we're using (e.g. `gpt-4`).
* `--agent` refers to which elicitation method we use to query the simulated human, among `questions_open` (generating open-ended questions), `questions_yn` (generating yes-or-no questions), `edge_cases` (generative active learning), `pool_diversity` (pool-based active learning with diversity sampling), `pool_random` (pool-based active learning with random sampling, used as a stand-in for supervised learning).
* `--eval_condition` refers to how often we produce evaluate the intermediate results of each transcript, with `per_turn` meaning we evaluate the transcript after each turn, `per_minute` meaning we evaluate the transcript only after each minute of interaction, and `at_end` meaning we only evaluate the transcript at the very end.
* `--pool_diversity_num_clusters` refers to the number of clusters we use for pool-based active learning with diversity sampling.
* `--task` refers to which domain we are evaluating (content recommendation, moral reasoning, email validation).

