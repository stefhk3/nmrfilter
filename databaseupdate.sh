#!/bin/bash

echo "Starting database update.."

wget https://nmrshiftdb.nmr.uni-koeln.de/hose2.txt

if [ $? -eq 0 ]; then
    echo "Database dump downloaded successfully."
else
    echo "Failed to download the file."
    exit 1
fi



java -jar DumpParser2-1.4.jar hose2.txt nmrshiftdbh.csv nmrshiftdbc.csv

jar uf simulate.jar nmrshiftdbh.csv nmrshiftdbc.csv

echo "Reference database inside Simulate.jar updated"

rm hose2.txt nmrshiftdbh.csv nmrshiftdbc.csv

#rm -rf temp_dir

exit 1

