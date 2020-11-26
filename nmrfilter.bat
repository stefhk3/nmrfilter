@echo off
python nmrfilter.py %1
java -cp "./*" uk.ac.dmu.simulate.Convert %1 > temp.txt
set /p OUT=<temp.txt
for /f "tokens=1,2,3 delims=_" %%a in ("%OUT%") do (
  set DL=%%a
  set SDFILE=%%b
  set PROJECTPATH=%%b
)
if /i %DL%=="true" (
    cd respredict
    python predict_standalone.py --filename %PROJECTPATH%%SDFILE% --format sdf --nuc 13C --sanitize --addhs false > %PROJECTPATH%predc.json
    python predict_standalone.py --filename %PROJECTPATH%%SDFILE% --format sdf --nuc 1H --sanitize --addhs false > %PROJECTPATH%predh.json
    cd ..
)
java -cp "./*" uk.ac.dmu.simulate.Simulate %1
python nmrfilter2.py %1