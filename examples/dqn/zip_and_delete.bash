#!/bin/bash

cd $1
# Remove forward slash
FILENAME=${2//\//}
START="$3"
EPISODE="$4"

# Browser-suppored compression
gzip $FILENAME
mv "$FILENAME.gz" "$FILENAME.js"
AWS_ACCESS_KEY_ID="$5" AWS_SECRET_ACCESS_KEY="$6" aws s3 cp \
    "$FILENAME.js" \
    "s3://aiworld/$START/$EPISODE" \
    --content-type "application/javascript" \
    --content-encoding "gzip"