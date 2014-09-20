#!/bin/bash

cd $1 && zip $2.zip $2 && rm "$1/$2"
