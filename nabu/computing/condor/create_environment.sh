#!/bin/sh

echo 'Host name is:'
echo $HOSTNAME

#create the necesary environment variables
source ~/.bashrc

#copy the ssh binary to enable ssh tunnels
cp /usr/bin/ssh /tmp


nvidia-smi  || true
{
meminfo=$(nvidia-smi | grep MiB | awk '{print $9,$11}' | sed 's/MiB//g')

meminfocommaseparated=""
for number in $meminfo
do
meminfocommaseparated="$meminfocommaseparated,$number"
done
} ||
{
meminfocommaseparated="None"
}

extrainput="--meminfo=$meminfocommaseparated"


# fullinput="$@ $extrainput"
fullinput="$@"

#run the original. Retry untill succes for a maximum of 5 runs
#for ind in 0 1 2 3 4
#do
#  $fullinput
#  echo 'Python script ended.'
#  ret=$?
#  echo $ret
#  if [ $ret -ne 0 ]; then
#    echo 'Error detected in python script. Rerun.'
#  else
#    echo 'Python script has ended succefully.'
#    break
#  fi
#done


echo 'command is:'
echo $fullinput

echo 'Running command'
echo '****'
echo ''
$fullinput