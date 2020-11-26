#!/bin/bash
python3 nmrfilter.py $1
out=$(java -cp "./*" uk.ac.dmu.simulate.Convert $1) 
#out=$(echo $out | tr -d '\n')
readarray -d _ -t outs <<<"$out"
echo "_${outs[2]}_"
if [ ${outs[0]} = 'true' ]
then
    cd respredict
    python3 predict_standalone.py --filename ${outs[2]}$1/${outs[1]} --format sdf --nuc 13C --sanitize --addhs false > ../../nmrfilterprojects/$1/predc.json
    #${outs[2]}/$1
    python3 predict_standalone.py --filename ${outs[2]}$1/${outs[1]} --format sdf --nuc 1H --sanitize --addhs false > ../../nmrfilterprojects/$1/predh.json
    cd ..
fi
java -cp "./*" uk.ac.dmu.simulate.Simulate $1
python3 nmrfilter2.py $1
