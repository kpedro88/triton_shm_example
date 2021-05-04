# Triton shared memory example

## Setup

```bash
wget https://raw.githubusercontent.com/kpedro88/triton_shm_example/master/setup.sh
chmod +x setup.sh
./setup.sh
cd CMSSW_11_3_0_pre5/src/HeterogeneousCore/SonicTriton/test/triton_shm_example
cmsenv
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
rm /dev/shm/shm*
cmsTriton -P -1 -v -n triton_server_instance -M $CMSSW_BASE/src/HeterogeneousCore/SonicTriton/data/models/ stop
```

## Results

`resnet_grpc_shm_client2.exe` works as intended and produces the following output:
```
shm : total = 3010560 (0x7fb508382000)
memcpy() : 602112 bytes, 2408448 remaining (0x7fb508415000)
memcpy() : 602112 bytes, 1806336 remaining (0x7fb5084a8000)
memcpy() : 602112 bytes, 1204224 remaining (0x7fb50853b000)
memcpy() : 602112 bytes, 602112 remaining (0x7fb5085ce000)
memcpy() : 602112 bytes, 0 remaining (0x7fb508661000)
batch 0
   inputs: 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, ...
   outputs: 0.000167075, 0.00060421, 7.23332e-05, 4.86887e-05, 0.000119915, 0.000246135, 1.82023e-05, 0.000186773, 5.15053e-05, 0.000431815, ...
batch 1
   inputs: 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, ...
   outputs: 0.000167215, 0.000612563, 7.18206e-05, 4.76253e-05, 0.000121269, 0.000249971, 1.84608e-05, 0.000179769, 5.15851e-05, 0.000418512, ...
batch 2
   inputs: 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, ...
   outputs: 0.000174489, 0.000615822, 7.38812e-05, 4.76297e-05, 0.000126701, 0.000263326, 1.94412e-05, 0.00017864, 5.38571e-05, 0.000416091, ...
batch 3
   inputs: 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, ...
   outputs: 0.000185543, 0.000615438, 7.74539e-05, 4.83911e-05, 0.000134673, 0.000277013, 2.10373e-05, 0.000179367, 5.68413e-05, 0.000421287, ...
batch 4
   inputs: 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ...
   outputs: 0.000197231, 0.00060928, 8.12652e-05, 4.95069e-05, 0.00014306, 0.00029046, 2.28279e-05, 0.000181557, 6.01525e-05, 0.000432174, ...
Shared Memory Status:
regions {
  key: "shm_input0"
  value {
    name: "shm_input0"
    key: "shm_input0"
    byte_size: 3010560
  }
}
regions {
  key: "shm_output0"
  value {
    name: "shm_output0"
    key: "shm_output0"
    byte_size: 20000
  }
}

PASS : System Shared Memory
```

`resnet_grpc_shm_client.exe` produces the same output using a nested vector approach, where all inner vectors share a contiguous memory region:
```
makeShmResource: overhead = 160, content = 3010560, total = 3010720 (0x7f2e5835d000)
TritonShmResource::allocate() : 160 bytes, 3010560 remaining (0x7f2e5835d000)
TritonShmResource::allocate() : 602112 bytes, 2408448 remaining (0x7f2e5835d0a0)
TritonShmResource::allocate() : 602112 bytes, 1806336 remaining (0x7f2e583f00a0)
TritonShmResource::allocate() : 602112 bytes, 1204224 remaining (0x7f2e584830a0)
TritonShmResource::allocate() : 602112 bytes, 602112 remaining (0x7f2e585160a0)
TritonShmResource::allocate() : 602112 bytes, 0 remaining (0x7f2e585a90a0)
offset0 = 160
batch 0
        inputs: 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, ...
        outputs: 0.000167075, 0.00060421, 7.23332e-05, 4.86887e-05, 0.000119915, 0.000246135, 1.82023e-05, 0.000186773, 5.15053e-05, 0.000431815, ...
batch 1
        inputs: 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, ...
        outputs: 0.000167215, 0.000612563, 7.18206e-05, 4.76253e-05, 0.000121269, 0.000249971, 1.84608e-05, 0.000179769, 5.15851e-05, 0.000418512, ...
batch 2
        inputs: 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, ...
        outputs: 0.000174489, 0.000615822, 7.38812e-05, 4.76297e-05, 0.000126701, 0.000263326, 1.94412e-05, 0.00017864, 5.38571e-05, 0.000416091, ...
batch 3
        inputs: 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, ...
        outputs: 0.000185543, 0.000615438, 7.74539e-05, 4.83911e-05, 0.000134673, 0.000277013, 2.10373e-05, 0.000179367, 5.68413e-05, 0.000421287, ...
batch 4
        inputs: 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ...
        outputs: 0.000197231, 0.00060928, 8.12652e-05, 4.95069e-05, 0.00014306, 0.00029046, 2.28279e-05, 0.000181557, 6.01525e-05, 0.000432174, ...
Shared Memory Status:
regions {
  key: "shm_input0"
  value {
    name: "shm_input0"
    key: "shm_input0"
    byte_size: 3010720
  }
}
regions {
  key: "shm_output0"
  value {
    name: "shm_output0"
    key: "shm_output0"
    byte_size: 20000
  }
}

TritonShmResource::deallocate() : 602112 bytes, 2408608 remaining
TritonShmResource::deallocate() : 602112 bytes, 1806496 remaining
TritonShmResource::deallocate() : 602112 bytes, 1204384 remaining
TritonShmResource::deallocate() : 602112 bytes, 602272 remaining
TritonShmResource::deallocate() : 602112 bytes, 160 remaining
TritonShmResource::deallocate() : 160 bytes, 0 remaining
PASS : System Shared Memory 
```
