python3 nmrfilter.py %1
java -cp "./*" uk.ac.dmu.simulate.Convert %1 > temp.txt
set /p OUT=<temp.txt
IF "%OUT%"=="true" (
    cd respredict
    conda activate respredict
    python predict_standalone.py --filename ../../nmrfilterprojects/artificial/testall.sdf --format sdf --nuc 13C --sanitize --addhs false > ../../nmrfilterprojects/artificial/predc.json
    python predict_standalone.py --filename ../../nmrfilterprojects/artificial/testall.sdf --format sdf --nuc 1H --sanitize --addhs false > ../../nmrfilterprojects/artificial/predh.json
    conda deactivate
    cd ..
)
java -cp "./*" uk.ac.dmu.simulate.Simulate %1
python3 nmrfilter2.py %1