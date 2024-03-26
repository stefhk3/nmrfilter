#!/bin/bash


echo "
           __   _                            __ 
|\ | |\/| |__) (_ . | |_  _  _       /|     |_  
| \| |  | | \  |  | | |_ (- |    \/   | .   __)                                                                                                        
"
# Get the directory path where the shell script is located
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


if [[ -z $CONDA_SHLVL || $CONDA_SHLVL == 0 ]]; then
    echo "Conda environment not in use, using Python Virtual Environment"
    # Check if the virtual environment already exists in the script's directory
    if [ -f "$script_dir/nmrfilter_env/bin/activate" ]; then
        echo "NMRfilter_env already exists. Activating..."
        source "$script_dir/nmrfilter_env/bin/activate"
    else
        echo "Creating virtual environment in $script_dir/nmrfilter_env..."
        python3 -m venv "$script_dir/nmrfilter_env"
        source "$script_dir/nmrfilter_env/bin/activate"

        echo "Installing the requirements..."
        python -m pip install -r requirements.txt

        echo "Installing Jupyter Notebook..."
        python -m pip install jupyter
    fi
else
    echo "Using conda environment.."
fi


#start the processing
python3 nmrfilter.py $1
out=$(java -cp "./*" uk.ac.dmu.simulate.Convert $1)

#out=$(echo $out | tr -d '\n')
readarray -d _ -t outs <<<"$out"
outs[2]=$(echo ${outs[2]} | tr -d '\n')


if [ ${outs[0]} = 'true' ]
then
    cd respredict
    python3 predict_standalone.py --filename ${outs[2]}$1/${outs[1]} --format sdf --nuc 13C --sanitize --addhs false > ${outs[2]}$1/predc.json
    python3 predict_standalone.py --filename ${outs[2]}$1/${outs[1]} --format sdf --nuc 1H --sanitize --addhs false > ${outs[2]}$1/predh.json
    cd ..
fi
java -cp "./*" uk.ac.dmu.simulate.Simulate $1
python3 nmrfilter2.py $1
