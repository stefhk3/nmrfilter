#!/bin/bash
python3 nmrfilter.py $1
out=$(java -cp "./*" uk.ac.dmu.simulate.Convert $1)
if [ $out = 'true' ]
then
    cd respredict
    conda activate respredict
    python predict_standalone.py --filename ../../nmrfilterprojects/$1/testall.sdf --format sdf --nuc 13C --sanitize --addhs false > ../../nmrfilterprojects/$1/predc.json
    python predict_standalone.py --filename ../../nmrfilterprojects/$1/testall.sdf --format sdf --nuc 1H --sanitize --addhs false > ../../nmrfilterprojects/$1/predh.json
    conda deactivate
    cd ..
fi
java -cp "./*" uk.ac.dmu.simulate.Simulate $1
python3 nmrfilter2.py $1
