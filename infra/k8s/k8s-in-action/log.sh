#!/bin/sh
set -eux

NAMESPACE=$1
DATE=$(date '+%Y%m%d%H%M%S')

kubectl describe pods -n "$NAMESPACE" > "./logs/$NAMESPACE.$DATE.log"
