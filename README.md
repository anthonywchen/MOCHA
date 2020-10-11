# MOCHA
This repository contains the data and code for the paper `MOCHA: A Dataset for Training and Evaluating Reading Comprehension Metrics.`

Paper: https://arxiv.org/abs/2010.03636

Website: https://allennlp.org/mocha

#### Setting up the environment
```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Baseline n-gram metrics
pip install git+https://github.com/salaniz/pycocoevalcap
```

#### Getting MOCHA
The MOCHA dataset is available at `data/mocha.tar.gz`. 

To reproduce the statistics of MOCHA found in the paper, first uncompress MOCHA, then run `python print_mocha_statistics.py`.

To check the SHA-1 hash of a data file on a Linux machine, run `sha1sum [filename]` and compare against the file `[filename].sha1`. 


#### Training LERC
Coming soon.