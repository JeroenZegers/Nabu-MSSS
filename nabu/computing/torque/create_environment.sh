#!/bin/sh

echo 'Host name is:'
echo $HOSTNAME


#create the necesary environment variables
source ~/.bashrc

nvidia-smi

fullinput="$@"
echo 'command is:'
echo $fullinput

echo 'Running command'
echo '****'
echo ''
$fullinput

