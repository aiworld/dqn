#!/bin/bash

# Remove forward slash
FILENAME=${2//\//}

cd $1 && zip $FILENAME.zip $FILENAME && rm "$1/$FILENAME"
