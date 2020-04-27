

all: paperfigs bib

bib:
	cp ~/Documents/Bibtex/NMR.bib ~/Dropbox/Apps/Overleaf/respredict/jonas_nmr.bib

paperfigs:
	cp "notebooks/model validate many_nuc uncertainty_13C_0.95.pdf"  ~/Dropbox/Apps/Overleaf/respredict/
	cp "notebooks/model validate many_nuc uncertainty_1H_0.95.pdf"  ~/Dropbox/Apps/Overleaf/respredict/

	cp "notebooks/compare results.nn_vs_hose_frac.13C.pdf"  ~/Dropbox/Apps/Overleaf/respredict/
	cp "notebooks/compare results.nn_vs_hose_frac.1H.pdf"  ~/Dropbox/Apps/Overleaf/respredict/

	cp "notebooks/method_compare_results.per_bond.13C.pdf"  ~/Dropbox/Apps/Overleaf/respredict/
	cp "notebooks/method_compare_results.per_bond.1H.pdf"  ~/Dropbox/Apps/Overleaf/respredict/

	cp "notebooks/NMRShiftDB stats and variability.molsize.pdf"  ~/Dropbox/Apps/Overleaf/respredict/


	cp "notebooks/methods_compare_results.all.13C.MAE.pdf"  ~/Dropbox/Apps/Overleaf/respredict/
	cp "notebooks/methods_compare_results.all.1H.MAE.pdf"  ~/Dropbox/Apps/Overleaf/respredict/


wordcount:
	detex ~/Dropbox/Apps/Overleaf/respredict/body.tex | wc -w

combined_pdf:
	"/System/Library/Automator/Combine PDF Pages.action/Contents/Resources/join.py" -o "/Users/jonas/Dropbox/Spectra to Smiles/respredict.pdf" "/Users/jonas/Dropbox/Spectra to Smiles/respredict_main.pdf" "/Users/jonas/Dropbox/Spectra to Smiles/respredict_supp.pdf"

tarball:
	tar czvf source_and_data.tar.gz atom_features.py graph_conv_many_nuc_pred.py netutil.py molecule_features.py netdataio.py nets.py respredict_pipeline.py util.py  graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.1H.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.1.mol_dict.pickle Docker/README.md


