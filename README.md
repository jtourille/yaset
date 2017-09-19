# Yet Another SEquence Tagger - YASET

YASET is sequence tagger written in [Python](https://www.python.org/).
The main objective of this project is to provide an off-the-shelf tool for various [NLP](https://en.wikipedia.org/wiki/Natural_language_processing)-related sequence tagging tasks 
(e.g. [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) or 
[Part-of-Speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)).

## 1. Installation

Step 0. You need a **working Python 3.3+ environment**.

Step 1. **Clone the repository**
```bash
git clone git@github.com:jtourille/yaset.git
```

Step 2. **Move** into the newly created directory
```bash
cd yaset
```

Step 3. **Install the tool** into your python environment
```bash
pip install .
```

## 2. Input Data

Data must be formatted in a **tabular** fashion with:
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

## 3. USAGE

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
