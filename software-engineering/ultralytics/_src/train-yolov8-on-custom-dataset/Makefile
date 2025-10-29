include .env
export

.PHONY: train

train: ${DATASETS_DIR}/pothole_dataset_v8/README.txt
	uv run main.py

${DATASETS_DIR}/pothole_dataset_v8/README.txt:
	mkdir -p ${DATASETS_DIR}
	wget "https://www.dropbox.com/s/qvglw8pqo16769f/pothole_dataset_v8.zip?dl=1" -O ${DATASETS_DIR}/pothole_dataset_v8.zip
	unzip -q ${DATASETS_DIR}/pothole_dataset_v8.zip -d ${DATASETS_DIR}
	rm -f ${DATASETS_DIR}/pothole_dataset_v8.zip
