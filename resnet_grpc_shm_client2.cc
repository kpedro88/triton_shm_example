// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <memory_resource>
#include <vector>
#include <iostream>
#include <string>
#include "grpc_client.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;

  exit(1);
}

}  // namespace

//helper class for shared memory
class TritonShmResource : public std::pmr::memory_resource {
public:
  TritonShmResource(std::string name, size_t size);
  virtual ~TritonShmResource();
  uint8_t* addr() { return addr_; }
  void close();
private:
  //required interface
  void* do_allocate(std::size_t bytes, std::size_t alignment) override;
  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override;
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;
  //member variables
  std::string name_;
  size_t size_;
  size_t counter_;
  uint8_t* addr_;
};

//based on https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/shm_utils.cc
//very simplified allocator:
//the shared memory region is created when the memory resource is initialized
//allocate() and deallocate() just increment and decrement a counter that keeps track of position in shm region
//region is actually destroyed in destructor
TritonShmResource::TritonShmResource(std::string name, size_t size) : name_(name), size_(size), counter_(0), addr_(nullptr) {
  //get shared memory region descriptor
  int shm_fd = shm_open(name_.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
  if (shm_fd == -1)
    throw std::runtime_error("SharedMemoryError: unable to get shared memory descriptor for key: "+name_);

  //extend shared memory object
  int res = ftruncate(shm_fd, size_);
  if (res == -1)
    throw std::runtime_error("SharedMemoryError: unable to initialize shared memory key "+name_+" to requested size: "+std::to_string(size_));

  //map to process address space
  constexpr size_t offset(0);
  addr_ = (uint8_t*)mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if(addr_ == MAP_FAILED)
    throw std::runtime_error("SharedMemoryError: unable to map to process address space for shared memory key: "+name_);

  //close descriptor
  if(::close(shm_fd) == -1)
    throw std::runtime_error("SharedMemoryError: unable to close descriptor for shared memory key: "+name_);
}

void TritonShmResource::close() {
  //unmap
  int tmp_fd = munmap(addr_, size_);
  if (tmp_fd == -1)
    throw std::runtime_error("SharedMemoryError: unable to munmap for shared memory key: "+name_);

  //unlink
  int shm_fd = shm_unlink(name_.c_str());
  if (shm_fd == -1)
    throw std::runtime_error("SharedMemoryError: unable to unlink for shared memory key: "+name_);
}

TritonShmResource::~TritonShmResource() {
  //avoid throwing in destructor
  try {
    close();
  }
  catch (...) {}
}

void* TritonShmResource::do_allocate(std::size_t bytes, std::size_t alignment) {
  size_t old_counter = counter_;
  counter_ += bytes;
  if(counter_>size_)
    throw std::runtime_error("Attempt to allocate "+std::to_string(bytes)+" bytes in region with only "+std::to_string(size_-old_counter)+" bytes free");
  void* result = addr_ + old_counter;
  std::cout << "TritonShmResource::allocate() : " << bytes << " bytes, " << size_-counter_ << " remaining (" << result << ")" << std::endl;
  return result;
}

void TritonShmResource::do_deallocate(void* p, std::size_t bytes, std::size_t alignment) {
  if(bytes>counter_)
    throw std::runtime_error("Attempt to deallocate "+std::to_string(bytes)+" bytes in region with only "+std::to_string(counter_)+" bytes used");
  counter_ -= bytes;
  std::cout << "TritonShmResource::deallocate() : " << bytes << " bytes, " << counter_ << " remaining" << std::endl;
}

bool TritonShmResource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return dynamic_cast<const TritonShmResource*>(&other) != nullptr;
}

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8001");
  nic::Headers http_headers;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:H:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }

  std::string model_name = "resnet50_netdef";
  std::string model_version = "";

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<nic::InferenceServerGrpcClient> client;
  FAIL_IF_ERR(
      nic::InferenceServerGrpcClient::Create(&client, url, verbose),
      "unable to create grpc client");

  // Unregistering all shared memory regions for a clean
  // start.
  FAIL_IF_ERR(
      client->UnregisterSystemSharedMemory(),
      "unable to unregister all system shared memory regions");
  FAIL_IF_ERR(
      client->UnregisterCudaSharedMemory(),
      "unable to unregister all cuda shared memory regions");

  size_t batch_size = 5;
  std::vector<int64_t> shape{batch_size, 3, 224, 224};
  size_t input_size = shape[1]*shape[2]*shape[3];
  size_t input_byte_size = sizeof(float)*batch_size*input_size;
  size_t output_size = 1000;
  size_t output_byte_size = sizeof(float)*batch_size*output_size;

  // Initialize the inputs with the data.
  nic::InferInput* input0;

  FAIL_IF_ERR(
      nic::InferInput::Create(&input0, "gpu_0/data", shape, "FP32"),
      "unable to get INPUT0");
  std::shared_ptr<nic::InferInput> input0_ptr;
  input0_ptr.reset(input0);

  // create shared memory vectors for input0 and input1
  std::string iname0("shm_input0");
  auto iresource0 = std::make_shared<TritonShmResource>(iname0, input_byte_size);
  std::vector<std::vector<float>> input0_vec(batch_size);
  float* input0_shm = (float*)iresource0->addr();
  std::cout << "shm : total = " << input_byte_size << " (" << input0_shm << ")" << std::endl;

  // initialize to dummy values
  size_t byte_size_per_batch = sizeof(float)*input_size;
  for (int i = 0; i < batch_size; ++i) {
    input0_vec[i].reserve(input_size);
    input0_vec[i].assign(input_size, float(i+1)/10.f);
    std::memcpy(input0_shm + i*input_size, input0_vec[i].data(), byte_size_per_batch);
    std::cout << "memcpy() : " << byte_size_per_batch << " bytes, " << input_byte_size - (i+1)*byte_size_per_batch << " remaining (" << input0_shm + (i+1)*input_size << ")" << std::endl;
  }

  FAIL_IF_ERR(
      client->RegisterSystemSharedMemory(iname0, iname0, input_byte_size),
      "failed to register input0 shared memory region");
  FAIL_IF_ERR(
      input0_ptr->SetSharedMemory(
          iname0, input_byte_size, 0 /* offset */),
      "unable to set shared memory for INPUT0");

  // Generate the outputs to be requested.
  nic::InferRequestedOutput* output0;

  FAIL_IF_ERR(
      nic::InferRequestedOutput::Create(&output0, "gpu_0/softmax"),
      "unable to get OUTPUT0");
  std::shared_ptr<nic::InferRequestedOutput> output0_ptr;
  output0_ptr.reset(output0);

  // Create Output0 in Shared Memory
  std::string oname0("shm_output0");
  auto oresource0 = std::make_shared<TritonShmResource>(oname0, output_byte_size);
  float* output0_shm = (float*)oresource0->addr();
  FAIL_IF_ERR(
      client->RegisterSystemSharedMemory(
          oname0, oname0, output_byte_size),
      "failed to register output0 shared memory region");
  FAIL_IF_ERR(
      output0_ptr->SetSharedMemory(
          oname0, output_byte_size, 0 /* offset */),
      "unable to set shared memory for OUTPUT0");

  // The inference settings. Will be using default for now.
  nic::InferOptions options(model_name);
  options.model_version_ = model_version;

  std::vector<nic::InferInput*> inputs = {input0_ptr.get()};
  std::vector<const nic::InferRequestedOutput*> outputs = {output0_ptr.get()};

  nic::InferResult* results;
  FAIL_IF_ERR(
      client->Infer(&results, options, inputs, outputs, http_headers),
      "unable to run model");
  std::shared_ptr<nic::InferResult> results_ptr;
  results_ptr.reset(results);

  // Validate the results...
  for (size_t i = 0; i < batch_size; ++i){
    std::cout << "batch " << i << std::endl;
    std::cout << "\tinputs: ";
    for (size_t j = 0; j < 10; ++j){
      std::cout << input0_shm[i*input_size+j] << ", ";
    }
    std::cout << "..." << std::endl;
    std::cout << "\toutputs: ";
    for (size_t j = 0; j < 10; ++j){
      std::cout << output0_shm[i*output_size+j] << ", ";
    }
    std::cout << "..." << std::endl;
  }

  // Get shared memory regions active/registered within triton
  inference::SystemSharedMemoryStatusResponse status;
  FAIL_IF_ERR(
      client->SystemSharedMemoryStatus(&status),
      "failed to get shared memory status");
  std::cout << "Shared Memory Status:\n" << status.DebugString() << "\n";

  // Unregister shared memory
  FAIL_IF_ERR(
      client->UnregisterSystemSharedMemory(iname0),
      "unable to unregister shared memory input0 region");
  FAIL_IF_ERR(
      client->UnregisterSystemSharedMemory(oname0),
      "unable to unregister shared memory output0 region");

  // Cleanup shared memory
  iresource0.reset();
  oresource0.reset();

  std::cout << "PASS : System Shared Memory " << std::endl;

  return 0;
}
