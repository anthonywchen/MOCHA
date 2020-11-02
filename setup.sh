### Sets up a virtualenv and installs Python packages
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/salaniz/pycocoevalcap

### Download datasets for pretraining
mkdir -p data/pretraining/raw
cd data/pretraining/raw

# Download BoolQ from https://github.com/google-research-datasets/boolean-questions and place it into `pretrain_data/raw/boolq`
mkdir -p boolq

# Download MCTest
wget https://github.com/mcobzarenco/mctest/archive/master.zip
unzip master.zip
mv mctest-master/data/MCTest .
mv MCTest mctest
rm master.zip
rm -r mctest-master

# Download MultiRC
wget https://cogcomp.org/multirc/data/mutlirc-v2.zip
unzip mutlirc-v2.zip
mv splitv2 multirc
rm mutlirc-v2.zip

# Download RACE
wget http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz
tar -zxvf RACE.tar.gz
mv RACE race
rm RACE.tar.gz