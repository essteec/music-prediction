#!/bin/bash

END="_opus.webm"
FILENAME=$1$END

ffprobe -v error -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 "data/audio/pilot/$FILENAME" |
  awk '{printf "%.0f\n", $1 * 1000}'
