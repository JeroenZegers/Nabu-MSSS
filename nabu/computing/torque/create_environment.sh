#!/bin/sh

echo 'Host name is:'
echo $HOSTNAME

echo 'Job ID is:'
echo $PBS_JOBID
fullinput="$@"

#create the necesary environment variables
source ~/.bashrc

nvidia-smi

echo 'copying dataset to local scratch and adjusting database config files'

TARGET_STRING="$VSC_DATA/dataforTF/MERL_segmented"
NEW_STRING="$VSC_SCRATCH_NODE/MERL_segmented"
file="$expdir/database.cfg"
grep -q $TARGET_STRING $file
if [ $? -eq 0 ]
then
    sed -i "s#${TARGET_STRING}#${NEW_STRING}#g" $file
else
    echo "$file does not contain the the target string"
    exit 1
fi

echo "1"

file=$expdir/*/database.cfg
grep -q $TARGET_STRING $file
if [ $? -eq 0 ]
then
    sed -i "s#${TARGET_STRING}#${NEW_STRING}#g" $file
else
    echo "$file does not contain the the target string"
    exit 1
fi

rsync -a $VSC_DATA/dataforTF/MERL_segmented $VSC_SCRATCH_NODE/. || exit 1

echo 'command is:'
echo $fullinput

echo 'Running command'
echo '****'
echo ''
$fullinput

echo 'reverting database config files to original'
sed -i "s#$VSC_SCRATCH_NODE/MERL_segmented#$VSC_DATA/dataforTF/MERL_segmented#g" $expdir/database.cfg
sed -i "s#$VSC_SCRATCH_NODE/MERL_segmented#$VSC_DATA/dataforTF/MERL_segmented#g" $expdir/*/database.cfg

