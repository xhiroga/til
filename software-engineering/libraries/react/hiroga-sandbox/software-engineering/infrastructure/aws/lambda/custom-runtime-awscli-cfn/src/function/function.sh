function handler() {
  aws --version
  EVENT_DATA=$1
  echo "$EVENT_DATA" 1>&2
  RESPONSE="Echoing request: '$EVENT_DATA'"

  echo $RESPONSE
}
