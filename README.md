# Triton shared memory example

## Setup

```bash
wget https://raw.githubusercontent.com/kpedro88/triton_shm_example/master/setup.sh
chmod +x setup.sh
./setup.sh
```

## Compiling

```bash
./compile.sh resnet_grpc_shm_client2.cc
./compile.sh resnet_grpc_shm_client.cc
```

## Running

```bash
cmsTriton -P -1 -v -n triton_server_instance -M $CMSSW_BASE/src/HeterogeneousCore/SonicTriton/data/models/ start
./resnet_grpc_shm_client2.exe
./resnet_grpc_shm_client.exe
cmsTriton -P -1 -v -n triton_server_instance -M $CMSSW_BASE/src/HeterogeneousCore/SonicTriton/data/models/ stop
```
