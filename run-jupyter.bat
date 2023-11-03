@echo off
setlocal

rem Get the directory path where the batch file is located
set "script_dir=%~dp0"

rem Check if the virtual environment already exists in the script's directory
if exist "%script_dir%\nmrfilter_env\Scripts\activate.bat" (
    echo NMRfilter_env already exists. Activating...
    call "%script_dir%\nmrfilter_env\Scripts\activate.bat"
) else (
    echo Creating virtual environment in "%script_dir%\nmrfilter_env"...
    python -m venv "%script_dir%\nmrfilter_env"
    call "%script_dir%\nmrfilter_env\Scripts\activate.bat"

    echo Installing the requirements...
    call python -m pip install -r requirements.txt
    
    echo Installing Jupyter Notebook...
    call python -m pip install jupyter
)

rem Start Jupyter Notebook
call jupyter notebook "%script_dir%\nmrfilter.ipynb"

pause
