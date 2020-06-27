#!/bin/bash
python3 nmrfilter.py $1
out=$(java -cp "./*" uk.ac.dmu.simulate.Convert $1)
readarray -d _ -t outs <<<"$out"
if [ ${outs[0]} = 'true' ]
then
    cd respredict
    python3 predict_standalone.py --filename ../../nmrfilterprojects/$1/${outs[1]} --format sdf --nuc 13C --sanitize --addhs false > ../../nmrfilterprojects/$1/predc.json
    python3 predict_standalone.py --filename ../../nmrfilterprojects/$1/${outs[1]} --format sdf --nuc 1H --sanitize --addhs false > ../../nmrfilterprojects/$1/predh.json
    cd ..
fi
java -cp "./*" uk.ac.dmu.simulate.Simulate $1
python3 nmrfilter2.py $1
