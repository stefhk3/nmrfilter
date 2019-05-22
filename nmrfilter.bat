@echo off
FOR /F "tokens=1,2 delims==" %%G IN (nmrproc.properties) DO (set %%G=%%H) 
if exist del %predictionoutput%
if exist del %clusteringoutput%
if exist del %louvainoutput%
del plots\*.png
echo "Simulating spectra for your compounds..."

java -cp "./*" uk.ac.dmu.simulate.Simulate

echo "Clustering the peaks in the measured spectrum..."

python3 clustering.py
echo "Detecting communities in the measures spectrum..."

python3 clusterlouvain.py

echo "Calculating best hits in your compounds..."
python3 similarity.py

pause
