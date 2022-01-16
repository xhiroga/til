PUBSUB=$1
gcloud pubsub topics publish ${PUBSUB} --message "hello"
