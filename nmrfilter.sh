echo "Simulating spectra for your compounds..."
java -cp "./*" uk.ac.dmu.simulate.Simulate
echo "Clustering the peaks in the measured spectrum..."
python3 clustering.py
echo "Detecting communities in the measures spectrum..."
python3 clusterlouvain.py
echo "Calculating best hits in your compounds..."
python3 similarity.py
