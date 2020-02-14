#!/usr/bin/env python
"""
Submit predictions of a CodeSearchNet model.

Usage:    
    predict.py -r RUN_ID     [-p PREDICTIONS_CSV] [-l LANGUAGES]
    predict.py -h | --help

Options:
    -h --help                       Show this screen
    -r, --wandb_run_id RUN_ID       wandb run ID, [username]/codesearchnet/[hash string id], viewable from run overview page via info icon
    -l, --languages LANGUAGES       Languages to evaluate
                                    [default: python,go,javascript,java,php,ruby]

Examples:
    ./submit_predict.py -r username/codesearchnet/0123456
"""

"""

"""

import pickle
import re
import shutil
import sys

from annoy import AnnoyIndex
from docopt import docopt
from dpu_utils.utils import RichPath
import pandas as pd
from tqdm import tqdm
import wandb
from wandb.apis import InternalApi

from dataextraction.python.parse_python_data import tokenize_docstring_from_string
import model_restore_helper

def get_csv_path(run, language):
    run_id = run.split("/")[-1]
    return '../resources/model_predictions/{}/model_predictions_{}.csv'.format(run_id, language)

def get_csv_path_final(run):
    run_id = run.split("/")[-1]
    return '../resources/model_predictions/{}/model_predictions.csv'.format(run_id)

if __name__ == '__main__':
    args = docopt(__doc__)

    args_wandb_run_id = args.get('--wandb_run_id')
    languages = args.get('--languages').split(",")

    result_df = pd.DataFrame([], columns=['query', 'language', 'identifier', 'url'])

    for language in languages:
        df = pd.read_csv(get_csv_path(args_wandb_run_id, language))
        result_df = result_df.append(df)

    predictions_csv = get_csv_path_final(args_wandb_run_id)
    result_df.to_csv(predictions_csv, index=False)
    
    print('Uploading predictions to W&B')
    # upload model predictions CSV file to W&B

    # we checked that there are three path components above
    entity, project, name = args_wandb_run_id.split('/')

    # make sure the file is in our cwd, with the correct name
    predictions_base_csv = "model_predictions.csv"
    shutil.copyfile(predictions_csv, predictions_base_csv)

    # Using internal wandb API. TODO: Update when available as a public API
    internal_api = InternalApi()
    internal_api.push([predictions_base_csv], run=name, entity=entity, project=project)
