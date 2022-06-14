#!/usr/bin/env bash

protoDir="../pytorch_util/protos"
outDir="../pytorch_util/pb"

python3 -m grpc_tools.protoc -I ${protoDir}/  --python_out=${outDir} --grpc_python_out=${outDir} ${protoDir}/*.proto
# python -m grpc_tools.protoc -I ./protos/  --python_out=./pb --grpc_python_out=./pb ./protos/*proto

