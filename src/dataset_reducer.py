#!/usr/bin/env python
"""
Remove near duplicates from data and perform train/test/validation/holdout splits.

Usage:
    dataset_reducer.py INPUT_FOLDERNAME OUTPUT_FOLDER RATE

Arguments:
    INPUT_FOLDER               directory w/ compressed jsonl files that have a .jsonl.gz a file extension
    OUTPUT_FOLDER              directory where you want to save data to.
    RATE FLOAT

Example:

    python dedup_split.py \
    --azure-info /ds/hamel/azure_auth.json \
    azure://semanticcodesearch/pythondata/raw_v2  \
    azure://semanticcodesearch/pythondata/Processed_Data_v2

"""

from docopt import docopt
import hashlib
import pandas as pd
from utils.pkldf2jsonl import chunked_save_df_to_jsonl
from dpu_utils.codeutils.deduplication import DuplicateDetector
import os
from tqdm import tqdm
from dpu_utils.utils import RichPath
from pathlib import Path
from utils.pkldf2jsonl import chunked_save_df_to_jsonl
import shutil

def jsonl_to_df(input_folder: RichPath) -> pd.DataFrame:
    "Concatenates all jsonl files from path and returns them as a single pandas.DataFrame ."

    assert input_folder.is_dir(), 'Argument supplied must be a directory'
    dfs = []
    files = list(input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'
    print(f'reading files from {input_folder.path}')
    for f in tqdm(files, total=len(files)):
        dfs.append(pd.DataFrame(list(f.read_as_jsonl(error_handling=lambda m,e: print(f'Error while loading {m} : {e}')))))
    return pd.concat(dfs)



def label_folds(df: pd.DataFrame, train_ratio: float, valid_ratio: float, test_ratio: float, holdout_ratio: float) -> pd.DataFrame:
    "Adds a partition column to DataFrame with values: {train, valid, test, holdout}."
    assert abs(train_ratio + valid_ratio + test_ratio + holdout_ratio - 1) < 1e-5,  'Ratios must sum up to 1.'
    # code in the same file will always go to the same split
    df['hash_key'] = df.apply(lambda x: f'{x.repo}:{x.path}', axis=1)
    df['hash_val'] = df['hash_key'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (2**16))

    train_bound = int(2**16 * train_ratio)
    valid_bound = train_bound + int(2**16 * valid_ratio)
    test_bound = valid_bound + int(2**16 * test_ratio)

    def label_splits(hash_val: int) -> str:
        if hash_val <= train_bound:
            return "train"
        elif hash_val <= valid_bound:
            return "valid"
        elif hash_val <= test_bound:
            return "test"
        else:
            return "holdout"

    # apply partition logic
    df['partition'] = df['hash_val'].apply(lambda x: label_splits(x))
    # display summary statistics
    counts = df.groupby('partition')['repo'].count().rename('count')
    summary_df = pd.concat([counts, (counts / counts.sum()).rename('pct')], axis=1)
    print(summary_df)

    return df


def load_concat_df(file_list):
    columns = ['repo', 'path', 'url', 'code', 
                     'code_tokens', 'docstring', 'docstring_tokens', 
                     'language', 'partition']
    return pd.concat([pd.read_json(f, 
        orient='records', 
        compression='gzip',
        lines=True)[columns] 
        for f in file_list], sort=False) 

def run(args):

    azure_info_path = args.get('--azure-info', None)
    input_folder = args['INPUT_FOLDERNAME']
    output_folder = RichPath.create(args['OUTPUT_FOLDER'])
    rate = float(args['RATE'])
    for language in ['python', 'go', 'javascript', 'java', 'ruby', 'php']:
        language_trainings_files = sorted(Path(f'{input_folder}/{language}/final/jsonl/train').glob('**/*.gz'))
        df = load_concat_df(language_trainings_files)
        print(df.count())
        sample = df.sample(frac = rate)
        print(sample.count())
        lang_output_folder = output_folder.join(f'{language}/final/jsonl/train')
        
        os.makedirs(str(lang_output_folder.to_local_path()))
        chunked_save_df_to_jsonl(df, lang_output_folder, basefilename=f'{language}_train')

        shutil.copytree(f'{input_folder}/{language}/final/jsonl/test', f'{output_folder}/{language}/final/jsonl/test')
        shutil.copytree(f'{input_folder}/{language}/final/jsonl/valid', f'{output_folder}/{language}/final/jsonl/valid')
        print(df.count())
        print(language_trainings_files)


    # # get data and process it
    # df = jsonl_to_df(f'{input_path}')
    # print('Removing fuzzy duplicates ... this may take some time.')
    # df = df.sample(frac=1, random_state=20181026)  # shuffle order of files
    # df = label_folds(df, train_ratio=train, valid_ratio=valid, test_ratio=test, holdout_ratio=holdout)
    # splits = ['train', 'valid', 'test', 'holdout']

    # for split in splits:
    #     split_df = df[df.partition == split]

    #     # save dataframes as chunked jsonl files
    #     jsonl_save_folder = output_folder.join(f'jsonl/{split}')
    #     print(f'Uploading data to {str(jsonl_save_folder)}')
    #     chunked_save_df_to_jsonl(split_df, jsonl_save_folder)

    #     # Upload dataframes to Azure
    #     filename = f'/tmp/{split}_df.pkl'
    #     df_save_path = output_folder.join(f'DataFrame/{split}_df.pkl')
    #     split_df.to_pickle(filename)
    #     print(f'Uploading data to {str(df_save_path)}')
    #     df_save_path.copy_from(RichPath.create(filename))
    #     os.unlink(filename)


if __name__ == '__main__':
    args = docopt(__doc__)
    run(args)
