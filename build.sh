TIMENOW=`date +%y.%m.%d.%H%M`
echo $TIMENOW
docker build -t pytorch_util_service:${TIMENOW} -f docker/Dockerfile .
