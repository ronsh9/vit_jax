VM_NAME=kmh-tpuvm-v3-128-2
ZONE=europe-west4-a

echo 'To kill jobs in: '$VM_NAME 'in' $ZONE' after 2s...'
sleep 2s

echo 'Killing jobs...'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
pgrep -af python | grep 'main.py' | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$1}' | sh
" # &> /dev/null
echo 'Killed jobs.'

# pgrep -af python | grep 'main.py' | grep -v 'grep' | awk '{print "sudo kill -9 " $1}' | sh

