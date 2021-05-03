#!/bin/bash

SRC=$1

g++ -l rt -std=c++17 \
-I /cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/triton-inference-server/2.3.0-ljfedo5/include -L /cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/triton-inference-server/2.3.0-ljfedo5/lib -l grpcclient \
-I /cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/protobuf/3.15.1-ljfedo/include -L /cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/protobuf/3.15.1-ljfedo/lib -l protobuf \
-I /cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/grpc/1.35.0-ljfedo2/include -L /cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/grpc/1.35.0-ljfedo2/lib -l grpc -l grpc++ \
-o $(basename $SRC .cc).exe $SRC

