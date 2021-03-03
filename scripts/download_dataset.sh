# !/bash/bin
DATA_URL=http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
wget --no-check-certificate ${DATA_URL}

tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
mv 1-billion-word-language-modeling-benchmark-r13output/ dataset/

/bin/rm 1-billion-word-language-modeling-benchmark-r13output.tar.gz
