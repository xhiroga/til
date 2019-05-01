PUBSUB=$1
gcloud beta functions deploy helloPubSub --trigger-resource ${PUBSUB} \
    --trigger-event google.pubsub.topic.publish \
    --runtime nodejs8
