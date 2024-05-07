@echo off
setlocal

:::
:::           __   _                            __
:::|\ | |\/| |__) (_ . | |_  _  _       /|     |_ 
:::| \| |  | | \  |  | | |_ (- |    \/   | .   __)
:::

for /f "delims=: tokens=*" %%A in ('findstr /b ::: "%~f0"') do @echo(%%A




rem Get the directory path where the batch file is located
set "script_dir=%~dp0"

if "%2" == "--update" (
    echo Updating the database..
    echo
    call ./databaseupdate.bat
    exit /b 0
)


rem Check if the virtual environment already exists in the script's directory


if "%CONDA_SHLVL%" == "" GOTO pyvenv
if "%CONDA_SHLVL%" == "0" GOTO pyvenv


echo Conda is activated, using conda environment..
GOTO :skip


:pyvenv
    echo Conda environment not in use, using Python Virtual Environment
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
:end

:skip
:end



rem Start processing
python nmrfilter.py %1
java -cp "simulate.jar;lib/*" uk.ac.dmu.simulate.Convert %1 > temp.txt
set /p OUT=<temp.txt
for /f "tokens=1,2,3 delims=_" %%a in ("%OUT%") do (
  set DL=%%a
  set SDFILE=%%b
  set PROJECTPATH=%%c
)

if %DL%==true (
    cd respredict
    python predict_standalone.py --filename %PROJECTPATH%%1\%SDFILE% --format sdf --nuc 13C --sanitize --addhs false > %PROJECTPATH%%1\predc.json
    python predict_standalone.py --filename %PROJECTPATH%%1\%SDFILE% --format sdf --nuc 1H --sanitize --addhs false > %PROJECTPATH%%1\predh.json
    cd ..
)

java -cp "simulate.jar;lib/*" uk.ac.dmu.simulate.Simulate %1
if "%2" == "--simulate" (
    python simulateplots.py %1
    echo Completed. Simulated spectra are available in the project directory.
) else (
    python nmrfilter2.py %1
)
