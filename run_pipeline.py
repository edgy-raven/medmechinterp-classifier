import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

import numpy as np
import tensorflow as tf

from collator_jobs import annotator, features, first_response, metadata, second_response, token_sequence
from pipeline_jobs import (
    dependency_features, embed_diagnosis, guided_accuracy, linguistic_features,
    lstm_autoencoder, simple_text_features, train_classifier, train_nn, transition_features
)
import openai_helper


job_order = [
    ("annotator", annotator),
    ("token_sequence", token_sequence),
    ("second_response", second_response),
    ("first_response", first_response),
    ("metadata", metadata),
    ("simple_text_features", simple_text_features),
    ("linguistic_features", linguistic_features),
    ("embed_diagnosis", embed_diagnosis),
    ("transition_features", transition_features),
    ("lstm_autoencoder", lstm_autoencoder),
    ("features", features),
    ("train", train_classifier),
    ("train_nn", train_nn),
    ("guided_accuracy", guided_accuracy),
]
featurizer_modules = {
    simple_text_features, 
    linguistic_features, 
    embed_diagnosis, 
    dependency_features, 
    lstm_autoencoder, 
    transition_features
}


if __name__ == "__main__":
    random.seed(7)
    np.random.seed(7)
    tf.random.set_seed(7)
    
    parser = argparse.ArgumentParser(description="Run the model training pipeline.")
    parser.add_argument(
        "--model",
        "--model-version",
        required=True,
    )
    parser.add_argument(
        "--input-dir",
        default="input_data",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=[
            "annotator",
            "token_sequence",
            "second_response",
            "first_response",
            "metadata",
            "simple_text_features",
            "linguistic_features",
            "embed_diagnosis",
            "transition_features",
            "lstm_autoencoder",
            "features",
            "train",
            "guided_accuracy",
        ],
        choices=list(job_module_pair[0] for job_module_pair in job_order),
        help="Jobs to run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run featurization even if outputs already exist.",
    )
    args, unknown_args = parser.parse_known_args()

    secrets = json.load(open("keyring.json"))
    openai_helper.initialize_client(secrets["openai_api_key"])

    requested_jobs = set(args.jobs or [])

    model_root = Path("output_artifacts") / args.model
    context = {
        "input_root": Path(args.input_dir),
        "output_root": model_root,
        "model_root": model_root,
        "feature_context": {},
        "user_context": unknown_args,
    }
    context["model_root"].mkdir(parents=True, exist_ok=True)

    for job_name, job_module in job_order:
        outputs = job_module.outputs(context)

        files_all_exist = all(path.exists() for path in outputs.values())
        in_pipeline = job_name in requested_jobs
        should_run_job = in_pipeline and (job_name == "train" or not files_all_exist or args.force)

        if in_pipeline and not should_run_job:
            print(f"Skipping {job_name}. Outputs already exist.")
        if should_run_job:
            job_module.run(**job_module.inputs(context))
        if files_all_exist or should_run_job:
            repository_to_update = context["feature_context"] if job_module in featurizer_modules else context
            repository_to_update.update(outputs)
