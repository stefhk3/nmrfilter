all: respredict.sif

respredict.sif: respredict.def
	sudo singularity build --force respredict.sif respredict.def
clean:
	rm respredict.sif
#deploy: respredict.sif
#	scp respredict.sif ejonas@cooley.alcf.anl.gov:
