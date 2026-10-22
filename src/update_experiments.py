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
update_experiments.py can recieve one of 3 arguments:
'test' -> Prints what was parsed from output/experiments.txt
'save' -> Saves experiment to experiments.csv
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
import subprocess

df = pd.read_csv('../results/experiments.csv')

exp_id = 0 if df.empty else df.iloc[-1]['experiment_id'] + 1

text = subprocess.run(
    'cat output/log.txt',
    shell=True,
    capture_output=True
)
experiments = subprocess.run(
    'cat output/experiments.txt',
    shell=True,
    capture_output=True
)
str_output = text.stdout.decode("utf-8")
exp_output = experiments.stdout.decode("utf-8")

info = exp_output.split('\n')
idx = info[0].find(':') + 2
author = info[0][idx:]

idx = info[1].find(':') + 2
date = info[1][idx:]

idx = info[2].find(':') + 2
model = info[2][idx:]

idx = info[3].find(':') + 2
features = info[3][idx:]

idx = info[4].find(':') + 2
standardization = info[4][idx:]

idx = info[5].find(':') + 2
augmentation = info[5][idx:]

idx = info[6].find(':') + 2
compression = info[6][idx:]

idx = info[6].find(':') + 2
compression = info[6][idx:]

notes_all = info[7:]
note_one_str = ''
for note in notes_all:
    note_one_str+= note + '\n'

idx = note_one_str.find(':') + 2
notes = note_one_str[idx:]


l1 = str_output.split('\n')[0]
idx = l1.find(':') + 2
accuracy = l1[idx:]

l2 = str_output.split('\n')[1]
idx = l2.find(':') + 2
model_size = l2[idx:]

l3 = str_output.split('\n')[2]
idx = l3.find(':') + 2
latency = l3[idx:]
if sys.argv[1] == 'test':
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
    new_row = {
    "experiment_id": exp_id,
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
    "notes": notes
    }
    df.loc[len(df)] = new_row
    df.to_csv('../results/experiments.csv', index=False)
    print(f"Experiment {exp_id} saved successfully!")
elif sys.argv[1] == 'remove':
    if exp_id != 1:
        df = pd.read_csv("../results/experiments.csv")
        df = df.iloc[:-1] 
        df.to_csv("../results/experiments.csv", index=False)
        print(f"Experiment {exp_id} removed successfully!")
    else:
        print('You are trying to remove the baseline experiment. Remove the if else (~line 170) if you truly intend that.')
else:
    print('Please supply the correct argument for this program (review update_experiments.py lines 32-36) for guidance.')
    







