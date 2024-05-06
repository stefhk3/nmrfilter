@echo off

echo Starting database update..

powershell -Command "Invoke-WebRequest https://nmrshiftdb.nmr.uni-koeln.de/hose2.txt -OutFile hose2.txt"

if %ERRORLEVEL% EQU 0 (
        echo Database dump downloaded successfully.
) else (
        echo Failed to download the file.
        exit /b 1
)


java -jar DumpParser2-1.4.jar hose2.txt nmrshiftdbh.csv nmrshiftdbc.csv

jar uf simulate.jar nmrshiftdbh.csv nmrshiftdbc.csv

echo Reference database inside Simulate.jar updated

del hose2.txt nmrshiftdbh.csv nmrshiftdbc.csv

exit /b 0