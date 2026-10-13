#!/bin/bash


echo "
           __   _                            __ 
|\ | |\/| |__) (_ . | |_  _  _       /|     |_  
| \| |  | | \  |  | | |_ (- |    \/   | .   __)                                                                                                        
"
# Get the directory path where the shell script is located
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ "$@" =~ "--update" ]]; then
    echo "Updating the database.."
    echo ""
    ./databaseupdate.sh
    exit 0
fi

if [[ -z $CONDA_SHLVL || $CONDA_SHLVL == 0 ]]; then
    echo "Conda environment not in use, using Python Virtual Environment"
    # Check if the virtual environment already exists in the script's directory
    if [ -f "$script_dir/nmrfilter_env/bin/activate" ]; then
        echo "NMRfilter_env already exists. Activating..."
        echo ""
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
    echo "Conda is activated, using conda environment.."
    echo ""
fi


#start the processing
python3 nmrfilter.py $1
# out=$(java -cp "./*" uk.ac.dmu.simulate.Convert $1)
out=$(java -cp "simulate.jar:lib/*" uk.ac.dmu.simulate.Convert $1)

outs=()
current=""
is_escaped=false

# Loop through the string character by character
for (( i=0; i<${#out}; i++ )); do
    char="${out:$i:1}"
    
    if [ "$is_escaped" = true ]; then
        if [ "$char" = "_" ]; then
            # Found an escaped underscore \_ -> keep the _ but drop the \
            current+="_"
        else
            # It was just a normal backslash followed by another character
            current+="\\$char"
        fi
        is_escaped=false
    elif [ "$char" = "\\" ]; then
        # Backslash detected, look at the next character in the next iteration
        is_escaped=true
    elif [ "$char" = "_" ]; then
        # Unescaped underscore -> delimiter found! Save element and reset
        outs+=("$current")
        current=""
    else
        current+="$char"
    fi
done

# Don't forget to add the final piece remaining in the buffer
if [ -n "$current" ] || [ "$char" = "_" ]; then
    outs+=("$current")
fi

outs[2]=$(echo ${outs[2]} | tr -d '\n')

if [ ${outs[0]} = 'true' ]
then
    cd respredict
    python3 predict_standalone.py --filename ${outs[2]}$1/${outs[1]} --format sdf --nuc 13C --sanitize --addhs false > ${outs[2]}$1/predc.json
    python3 predict_standalone.py --filename ${outs[2]}$1/${outs[1]} --format sdf --nuc 1H --sanitize --addhs false > ${outs[2]}$1/predh.json
    cd ..
fi

#java -cp "./*" uk.ac.dmu.simulate.Simulate $1 > /dev/null 2>&1
java -cp "simulate.jar:lib/*" uk.ac.dmu.simulate.Simulate $1 > /dev/null 2>&1

if [[ "$@" =~ "--simulate" ]]; then
    echo ""
    python3 simulateplots.py $1
    echo "Completed. Simulated spectra are available in the project directory."
else
    python3 nmrfilter2.py $1
fi