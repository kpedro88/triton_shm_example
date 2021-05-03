#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh

scram project CMSSW_11_3_0_pre5
cd CMSSW_11_3_0_pre5/src
eval `scramv1 runtime -sh`
git cms-addpkg HeterogeneousCore/SonicTriton
scram b
cd HeterogeneousCore/SonicTriton/test
./fetch_model.sh
git clone https://github.com/kpedro88/triton_shm_example

