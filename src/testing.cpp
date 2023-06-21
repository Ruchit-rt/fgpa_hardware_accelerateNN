#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <limits>
#include "dpc_common.hpp"
#include <oneapi/dpl/random>
#include <oneapi/mkl.hpp>
#include <cinttypes>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace std::chrono;
using namespace std;
using namespace sycl;

// Test case 1: Basic functionality
void test_quant(queue &q, int8_t *urop) {
    int chn = 3;
    int size = 3;
    int d = 1;
    
    // Input array
    int8_t input[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};
    
    // Filter array
    int8_t filter[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    
    // Biases array
    int32_t biases[1] = {10};
    
    // Expected output
    int8_t expected_output[9] = {
        24, 45, 48,
        67, -126, 112,
        120, -76, -124
    };
    int8_t * output = new int8_t[9];

    int8_t* input_ptr = (int8_t*) malloc_device(9 * sizeof(int8_t), q);
    auto input_to_device_event = q.memcpy(input_ptr, &input[0], 9 * sizeof(int8_t));
    input_to_device_event.wait();

    // Call the function
    int8_t* output_ptr = conv_pad_q(q, chn, size, input_ptr, filter, d, biases, 1, 1, 1);
    
    auto logits_to_host = q.submit([&] (handler &h) {
        h.memcpy(&output[0], output_ptr, 9 * sizeof(int8_t));
    });
    logits_to_host.wait();

    // Compare the result
    for (int i = 0; i < chn * size * size; i++) {
        assert(output[i] == expected_output[i]);
    }
    
    std::cout << "Test case 1 passed!" << std::endl;
}

int main(){

      // The file that encodes all parameters of the model.
    ifstream rf_data("data/model_params_quant.mmzk", ios::out | ios::binary);
    if(!rf_data) {
        cout << "Cannot open file!" << std::endl;
        return 1;
    }

    // Print out hardware name.
    cout << q.get_device().get_info<info::device::name>() << "\n";

    // Get parameters
    int8_t *weights1 = read_param_int8(rf_data);
    int32_t *biases1 = read_param_int32(rf_data);
    int8_t *weights2 = read_param_int8(rf_data);
    int32_t *biases2 = read_param_int32(rf_data);
    float *prototypes = read_param_float(rf_data);
    int8_t *fc_weights = read_param_int8(rf_data);

    rf_data.close();

      // Allocate memories for intermediate computations.
    int8_t *input = new int8_t[3 * 448 * 448];
    int8_t *conved1 = new int8_t[64 * 224 * 224];
    int8_t *pooled1 = new int8_t[64 * 112 * 112];
    int8_t *conved2 = new int8_t[512 * 112 * 112];
    int8_t *pooled2 = new int8_t[512 * 56 * 56];
    float *pooled2_f = new float[512 * 56 * 56];
    float *distances_f = new float[15 * 56 * 56];
    float *similarities_f = new float[15 * 56 * 56];
    float *avg_f = new float[15];
    int8_t *avg = (int8_t *)avg_f;
    int8_t *logits = new int8_t[3];
    float *logits_f = new float[3];
    float *upsampled_f = new float[15 * 224 * 224];


    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    queue q(selector);
    auto device = q.get_device();

    if (!device.has(sycl::aspect::usm_device_allocations)) {
      std::cerr << "This design must either target a board that supports USM "
                   "Host/Shared allocations, or IP Component Authoring. "
                << std::endl;
      std::terminate();
    }

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    quant(q, 3 * 224 * 224, 0.01979798823595047, input_f, input);

    // Convolutional layers (* 2).
    conv_pad_q(q, 3, 224, input, weights1, 64, biases1, 0.01979798823595047, 0.013484773226082325, 0.04881289601325989, conved1);
    max_pool_q(q, 64, 224, 224, conved1, pooled1);
    conv_pad_q(q, 64, 112, pooled1, weights2, 512, biases2, 0.04881289601325989, 0.0006907337228767574, 0.016132580116391182, conved2);
    max_pool_q(q, 512, 112, 112, conved2, pooled2);
    dequant(q, 512 * 56 * 56, 0.016132580116391182, pooled2, pooled2_f);

    // Prototype layer.
    l2_distance(q, 512, 56 * 56, pooled2_f, 15, prototypes, distances_f);
    distance_2_similarity(q, 15 * 56 * 56, distances_f, similarities_f);
    top9_average_pooling(q, 15, 56 * 56, similarities_f, avg_f, true);

    // Fully-connected layer.
    quant(q, 15, 0.01979798823595047, avg_f, avg);
    fully_connected(q, 15, 3, avg, fc_weights, logits);
    dequant(q, 3, 0.06617073714733124, logits, logits_f);

    // Compute upsampled activation map (information for interpretation).
    upsample4(q, 15, 56, 56, similarities_f, upsampled_f);

    test_quant(input);
    test_conv1(conved1);
    test_pool1(pooled1);
    test_conv2(conved2);
    test_pool2(pooled2_f);
    test_dequant(pooled2);
    test_sims(similarities_f);
    test_avg(avg);
    test_logits(logits);
    test_upsample(upsampled_f);    
    
    int8_t *input = new int8_t[3 * 448 * 448];
    int8_t *conved1 = new int8_t[64 * 224 * 224];
    int8_t *pooled1 = new int8_t[64 * 112 * 112];
    int8_t *conved2 = new int8_t[512 * 112 * 112];
    int8_t *pooled2 = new int8_t[512 * 56 * 56];
    float *pooled2_f = new float[512 * 56 * 56];
    float *distances_f = new float[15 * 56 * 56];
    float *similarities_f = new float[15 * 56 * 56];
    float *avg_f = new float[15];
    int8_t *avg = (int8_t *)avg_f;
    int8_t *logits = new int8_t[3];
    float *logits_f = new float[3];
    float *upsampled_f = new float[15 * 224 * 224];

    return 0;
}