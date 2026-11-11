"""Updates the Experiments.csv Folder for October Milestone"""

"""
Usage Tips:
LOG.TXT:
Create output/log.txt, by either running build_model.py on your model(s) of interest,
or following the format specified in build_model.py.
update_experiments.py extracts Accuracy, Model Size, and Mean Latency from output/log.txt

EXPERIMENTS.TXT:
Create output/experiments.txt 
You will log the remaining information for each experiment by writing in that file.

(As of now) you are responsible for logging (in experiments.txt):
Author, Date, Model, Features, Standardization, Augmentation, Compression, Notes
For all fields you are responsible for, EXCEPT 'Notes', you need to include the name of the field, 
followed by a semi-colon, followed by a space, then the content. (:[space] is the most important part there. )
For all fields you are responsible for, EXCEPT 'Notes', the field information should be contained on ONE line.
The 'Notes' field should include the name of the field, followed by a semi-colon, followed by a space, then the content.
The 'Notes' field CAN exist on multiple lines. 

i.e:
Author: Tule Achor (GOOD)
Author: Tule 
Achor (BAD)
Author Tule Achor (BAD)
Author:Tule Achor (BAD)
Notes: Tule
Achor (GOOD)


ARGUMENTS:
update_experiments.py can recieve one of 4 arguments:
'test' -> Prints what was parsed from output/experiments.txt
'save' -> Saves experiment to experiments.csv
'update' -> Overwrites the most recent experiment in experiments.csv (fixes a row instead of adding one)
'remove' -> Removes most recent experiment from experiments.csv

EXAMPLES:
A full example of logging experiments with log.txt and experiments.txt

log.txt:
Accuracy Score: 0.9803921568627451
File size (MB): 0.00496
Mean Latency: 0.17619294034375343
Latency stdev: 0.004968544210077892
Min Latency: 0.17120900156442076
Max Latency: 0.210749996767845
Sampled 100 times.

experiments.txt:
Author: Tule Achor
Date: 10/21/2026
Model: SVM
Features: ['rms_time', 'kurtosis_time', 'stdev_time', 'rms_four','kurtosis_four', 'stdev_four', 'rms_spec', 'kurtosis_spec','stdev_spec','rms_psd', 'kurtosis_psd','stdev_psd']
Standardization: Global z-score
Augmentation: None
Compression: None
Notes: Testing for some other functionalities (Still Baseline) 

(In src) 
To make sure it parsed correctly:
python3 update_experiments.py test

Once I'm happy everything parsed well:
I would then run python3 update_experiments.py save 

I made a mistake, and wanted to undo: 
I would then run python3 update_experiments.py remove 

Please shoot me (Tule) a message though IMessage if you have any questions/something isn't working/my docstring is trash...

"""
import pandas as pd
import sys

CSV_PATH = '../results/experiments.csv'

# Read everything as plain text and do NOT treat the word "None" as missing.
# (pandas' default turns "None" into an empty cell, which is how the baseline row
# ended up with blank Augmentation / Compression columns.)
df = pd.read_csv(CSV_PATH, dtype=object, keep_default_na=False)

# Next id for 'save'. 'update' re-uses the id of the last row instead.
exp_id = 0 if df.empty else int(df.iloc[-1]['experiment_id']) + 1


def read_text(path):
    """Read a text file (works on Windows, Mac, and Colab; no shell needed)."""
    with open(path, encoding='utf-8') as f:
        return f.read().replace('\r\n', '\n')


def after_colon(line):
    """Everything after the first ': ' on a line, with stray spaces removed."""
    return line.split(':', 1)[1].strip()


str_output = read_text('output/log.txt')
exp_output = read_text('output/experiments.txt')

info = exp_output.split('\n')
author = after_colon(info[0])
date = after_colon(info[1])
model = after_colon(info[2])
features = after_colon(info[3])
standardization = after_colon(info[4])
augmentation = after_colon(info[5])
compression = after_colon(info[6])

# Notes can span several lines: keep everything after the first ':' of the Notes line.
notes = after_colon('\n'.join(info[7:]))

log_lines = str_output.split('\n')
accuracy = float(after_colon(log_lines[0]))
model_size = float(after_colon(log_lines[1]))
latency = float(after_colon(log_lines[2]))


def build_row(experiment_id):
    return {
        "experiment_id": experiment_id,
        "date": date,
        "author": author,
        "model": model,
        "input_repr": features,
        "standardization": standardization,
        "augmentation": augmentation,
        "compression": compression,
        "accuracy_test": accuracy,
        "model_size_mb": model_size,
        "latency_ms_per_sample": latency,
        "notes": notes,
    }


if len(sys.argv) < 2:
    print("Please supply an argument: test, save, update, or remove (see the usage tips at the top of this file).")
elif sys.argv[1] == 'test':
    print('exp_id:', exp_id)
    print('author:', author)
    print('date: ', date)
    print('model: ', model)
    print('features: ', features)
    print('standardization:', standardization)
    print('augmentation: ', augmentation)
    print('compression: ', compression)
    print('accuracy: ', accuracy)
    print('model_size: ', model_size)
    print('latency: ', latency)
    print('notes: ', notes)
elif sys.argv[1] == 'save':
    df.loc[len(df)] = build_row(exp_id)
    df.to_csv(CSV_PATH, index=False)
    print(f"Experiment {exp_id} saved successfully!")
elif sys.argv[1] == 'update':
    # Overwrites the MOST RECENT row with what is in output/log.txt and
    # output/experiments.txt. Use this to fix a row instead of adding a new one.
    if df.empty:
        print('Nothing to update: experiments.csv has no rows yet. Use "save" instead.')
    else:
        last_id = int(df.iloc[-1]['experiment_id'])
        df.loc[df.index[-1]] = build_row(last_id)
        df.to_csv(CSV_PATH, index=False)
        print(f"Experiment {last_id} updated successfully!")
elif sys.argv[1] == 'remove':
    # Removes the MOST RECENT row (including the baseline, if it is the only row).
    if df.empty:
        print('Nothing to remove: experiments.csv has no rows.')
    else:
        last_id = int(df.iloc[-1]['experiment_id'])
        df = df.iloc[:-1]
        df.to_csv(CSV_PATH, index=False)
        print(f"Experiment {last_id} removed successfully!")
else:
    print('Please supply the correct argument: test, save, update, or remove (see the usage tips at the top of this file).')
    







