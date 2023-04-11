setup-env:
	mamba env create --file=environment.yml

setup-data:
	python src/ingest_data.py
	python src/write_train_test_split.py

process-cat:
	python src/process_cat.py label_encoding
	python src/process_cat.py one_hot_encoding
	python src/process_cat.py freq_encoding