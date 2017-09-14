# Yet Another SEquence Tagger

YASET is sequence tagger written in Python. For now, one model is implemented by several state-of-the-art models
 will be implemented in the near future.
The main objective of this tool is to be usable as an off-the-shelf tool for various sequence tagging tasks 
(e.g. Named Entity Recognition or Part-of-Speech tagging).

## Installation

0. You need a working Python 3.6+ environment.

1. Clone the repository
```text
git clone git@github.com:jtourille/yaset.git
```

2. Move to the newly created directory
```text
cd yaset
```

3. Install the tool into your python environment
```text
pip install .
```

## Input Data

Data must be formatted in a tabular fashion with:
* One token per line
* Sequence separated by blank lines

The first column must contain the tokens and the last column must contain the labels.
You can add as many other columns you want between these two columns. They will simply be ignored.

Data file example
```text
Detection   O
of  O
Nipah   O
virus   O
antibody    O
among   O
bat O
serum   B-Organism_substance
samples I-Organism_substance
collected   O
from    O
10  O
provinces   O
in  O
China   O
,   O
2004    O
-   O
2007    O
*   O

```

## How to use

The tool has two modes: TRAIN and APPLY.

### TRAIN

Once you have installed the tool and correctly formatted your input data, make a copy of the configuration file and 
adapt the values according to your situation.

```text
cp config.ini config-xp1.ini
```

Launch the tool by invoking the yaset command

```text
yaset LEARN --config /path/to/config/config-xp1.ini
```

## APPLY

Models learned by the tool take the form of directories. If you want to apply a learned model on your test data, run
 the following command
 
```text
yaset APPLY --working_dir /path/to/working/dir \
    --input_path /path/to/data/test.tab \
    --model_path /path/to/model/yaset-learn-YYYYMMDD-HHMMSS
```


