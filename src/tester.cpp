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

// Number of iterations.
#define N 5

template <std::size_t ID> class Conv1ID;
template <std::size_t ID> class Conv2ID;

void exception_handler(sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
    }
  }
}

// void print_exec_time(event e, string name){
//     auto start_time = e.template
//             get_profiling_info<sycl::info::event_profiling::command_start>();
//     auto end_time = e.template
//             get_profiling_info<sycl::info::event_profiling::command_end>();
//     double dur = (end_time - start_time) / 1.0e9;

//     cout <<  name << " event time: " << dur << std::endl;
// }

// Read int32 parameters from the given input stream.
int32_t *read_param_int32(ifstream &rf)
{
    int len;
    rf.read((char *)(&len), 4);
    int32_t *result = new int32_t[len];
    rf.read((char *)result, len * 4);
    return result;
}

// Read int8 parameters from the given input stream.
int8_t *read_param_int8(ifstream &rf)
{
    int len;
    rf.read((char *)(&len), 4);
    int8_t *result = new int8_t[len];
    rf.read((char *)result, len);
    return result;
}

// Read float (non-quantised) parameters from the given input stream.
float *read_param_float(ifstream &rf)
{
    int len;
    rf.read((char *)(&len), 4);
    float *result = new float[len];
    rf.read((char *)result, len * 4);
    return result;
}

template <template <std::size_t ID> typename Name, int chn, int size, int d>
// /* Carry out quantised convolution (with padding) between the TENSOR (1 * CHN * W * W) and the FILTER (D * CHN * 3 * 3), then ReLU. */
int8_t* conv_pad_q(queue &q, int8_t *tensor_ptr, int8_t *filter, int32_t *biases, float tensor_scale, float filter_scale, float result_scale)
{
    int8_t* filter_ptr = (int8_t*) malloc_device(d * chn * 9 * sizeof(int8_t), q);
    int32_t* bias_ptr = (int32_t*) malloc_device(d * sizeof(int32_t), q);
    int8_t* result_ptr = (int8_t*) malloc_device(d * size * size * sizeof(int8_t), q);

    auto filter_to_device_event = q.memcpy(filter_ptr, &filter[0], d * chn * 9 * sizeof(int8_t));
    auto bias_to_device_event = q.memcpy(bias_ptr, &biases[0], d * sizeof(int32_t));

    auto corner_event = q.submit([&](handler &h)
    {
        h.depends_on(filter_to_device_event);
        h.depends_on(bias_to_device_event);
        
        h.single_task<Name<0>>([=]() [[intel::kernel_args_restrict]]{
            device_ptr<int8_t> tensor_d(tensor_ptr);
            device_ptr<int8_t> filter_d(filter_ptr);
            device_ptr<int32_t> bias_d(bias_ptr);
            device_ptr<int8_t> result_d(result_ptr);

            for (int index = 0; index < d; index++){
                int32_t sum = 0;
                const float scale = tensor_scale * filter_scale / result_scale;
                for (int c = 0; c < chn; c++) {
                    int _fi = index * chn + c;
                    sum += filter_d[_fi * 9 + 4] * tensor_d[c * size * size];
                    sum += filter_d[_fi * 9 + 5] * tensor_d[c * size * size + 1];
                    sum += filter_d[_fi * 9 + 7] * tensor_d[c * size * size + size];
                    sum += filter_d[_fi * 9 + 8] * tensor_d[c * size * size + size + 1];
                }

                sum += bias_d[index];
                result_d[index * size * size] = sum > 0 ? round(sum * scale) : 0;

                sum = 0;
                for (int c = 0; c < chn; c++) {
                    int _fi = index * chn + c;
                    sum += filter_d[_fi * 9 + 3] * tensor_d[c * size * size + size - 2];
                    sum += filter_d[_fi * 9 + 4] * tensor_d[c * size * size + size - 1];
                    sum += filter_d[_fi * 9 + 6] * tensor_d[c * size * size + size + size - 2];
                    sum += filter_d[_fi * 9 + 7] * tensor_d[c * size * size + size + size - 1];
                }

                sum += bias_d[index];
                result_d[index * size *size + size - 1] = sum > 0 ? round(sum * scale) : 0;

                sum = 0;
                for (int c = 0; c < chn; c++) {
                    int _fi = index * chn + c;
                    sum += filter_d[_fi * 9 + 1] * tensor_d[c * size * size + (size - 2) * size];        
                    sum += filter_d[_fi * 9 + 2] * tensor_d[c * size * size + (size - 2) * size + 1];        
                    sum += filter_d[_fi * 9 + 4] * tensor_d[c * size * size + (size - 1) * size];        
                    sum += filter_d[_fi * 9 + 5] * tensor_d[c * size * size + (size - 1) * size + 1];        
                }

                sum += bias_d[index];
                result_d[index * size * size + (size - 1) * size] = sum > 0 ? round(sum * scale) : 0;
                
                sum = 0;
                for (int c = 0; c < chn; c++) {
                    int _fi = index * chn + c;
                    sum += filter_d[_fi * 9] * tensor_d[c * size * size + (size - 2) * size + size - 2];
                    sum += filter_d[_fi * 9 + 1] * tensor_d[c * size * size + (size - 2) * size + size - 1];
                    sum += filter_d[_fi * 9 + 3] * tensor_d[c * size * size + (size - 1) * size + size - 2];
                    sum += filter_d[_fi * 9 + 4] * tensor_d[c * size * size + (size - 1) * size + size - 1];
                }

                sum += bias_d[index];
                result_d[index * size * size + (size - 1) * size + size - 1] = sum > 0 ? round(sum * scale) : 0;
            }
        });
    });

    // Task for the boundary elements.
    auto boundry_event = q.submit([&](handler &h) {
        h.depends_on(filter_to_device_event);
        h.depends_on(bias_to_device_event);

        h.single_task<Name<1>>([=]() {
            device_ptr<int8_t> tensor_d(tensor_ptr);
            device_ptr<int8_t> filter_d(filter_ptr);
            device_ptr<int32_t> bias_d(bias_ptr);
            device_ptr<int8_t> result_d(result_ptr);

            const float scale = tensor_scale * filter_scale / result_scale;
            for (int i = 0; i < d * (size - 2); i++) {
                int index[2] = { i / (size - 2), i % (size - 2) };
                int32_t sum1 = 0;
                #pragma unroll
                for (int c = 0; c < chn; c++) {
                    int _fi = index[0] * chn + c;
                    #pragma unroll
                    for (int i = 0; i <= 1; i++) {
                        #pragma unroll
                        for (int j = 0; j <= 2; j++) {
                            sum1 += filter_d[_fi * 9 + (i+1) * 3 + j] * tensor_d[c * size * size + (i * size) + index[1] + j];
                        }
                    }
                }

                sum1 += bias_d[index[0]];
                result_d[index[0] * size * size + index[1] + 1] = sum1 > 0 ? round(sum1 * scale) : 0;

                int32_t sum2 = 0;
                for (int c = 0; c < chn; c++) {
                    int _fi = index[0] * chn + c;
                    #pragma unroll
                    for (int i = -2; i <= -1; i++) {
                        #pragma unroll
                        for (int j = 0; j <= 2; j++) {
                            sum2 += filter_d[_fi * 9 + (i + 2) * 3 + j] * tensor_d[c * size * size + (size + i) * size + index[1] + j];      
                        }
                    }
                }

                sum2 += bias_d[index[0]];
                result_d[index[0] * size * size + (size - 1) * size + index[1] + 1] = sum2 > 0 ? round(sum2 * scale) : 0;

                int32_t sum3 = 0;
                for (int c = 0; c < chn; c++) {
                    int _fi = index[0] * chn + c;
                    #pragma unroll
                    for (int i = 0; i <= 2; i++) {
                        #pragma unroll
                        for (int j = 0; j <= 1; j++) {
                            sum3 += filter_d[_fi * 9 + i * 3 + j + 1] * tensor_d[c * size * size + (index[1] + i) * size + j];        
                        }
                    }
                }

                sum3 += bias_d[index[0]];
                result_d[index[0] * size * size + (index[1] + 1) * size] = sum3 > 0 ? round(sum3 * scale) : 0;

                int32_t sum4 = 0;
                for (int c = 0; c < chn; c++) {
                    int _fi = index[0] * chn + c;
                    #pragma unroll
                    for (int i = 0; i <= 2; i++) {
                        #pragma unroll
                        for (int j = -2; j <= -1; j++) {
                        sum4 += filter_d[_fi * 9 + i * 3 + j + 2] * tensor_d[c * size * size + (index[1] + i) * size + size + j];        
                        }
                    }
                }

                sum4 += bias_d[index[0]];
                result_d[index[0] * size * size + (index[1] + 1) * size + size - 1] = sum4 > 0 ? round(sum4 * scale) : 0;
            }
        });
    });

    // Task for interior elements (that uses all 3 * 3 filters).
    auto interior_event = q.submit([&](handler &h) {
        h.depends_on(filter_to_device_event);
        h.depends_on(bias_to_device_event);

        h.single_task<Name<2>>([=]() {
            device_ptr<int8_t> inp_d(tensor_ptr);
            device_ptr<int8_t> filter_d(filter_ptr);
            device_ptr<int32_t> bias_d(bias_ptr);
            device_ptr<int8_t> result_d(result_ptr);

            const float scale = tensor_scale * filter_scale / result_scale;
            for (int i = 0; i < (size - 2) * (size - 2); i++){
                int index[3] = { i / ((size - 2) * (size - 2)), (i / (size -2)) % (size - 2), i % (size - 2) };
                int32_t sum = 0;
                #pragma unroll 2 // Partial unrolling for the outermost loop.
                for (int c = 0; c < chn; c++) {
                    int _fi = index[0] * chn + c;
                    for (int i = 0; i <= 2; i++) {
                        for (int j = 0; j <= 2; j++) {
                            sum += filter_d[index[0] * chn + c * 9 + i * 3 + j] * inp_d[c * size * size + (index[1] + i) * size + index[2] + j];        
                        }
                    }
                }

                sum += bias_d[index[0]];
                result_d[index[0] * size * size + (index[1] + 1) * size + index[2] + 1] = sum > 0 ? round(sum * scale) : 0;
            }
        });
    });

    corner_event.wait();
    boundry_event.wait();
    interior_event.wait();

    // print_exec_time(corner_event, "Corner conv");
    // print_exec_time(boundry_event, "Boundry conv");
    // print_exec_time(interior_event, "Interior conv");

    free(tensor_ptr, q);
    free(filter_ptr, q);
    free(bias_ptr, q);

    return result_ptr;
}

// Quantise the input TENSOR with the given SCALE.
int8_t* quant(queue &q, int size, float scale, float *tensor_ptr)
{
    int8_t* result_ptr = (int8_t*) malloc_device(size * sizeof(int8_t), q);

    auto quant_event = q.submit([&](handler &h) {
        h.single_task<class QuantID>([=]() [[intel::kernel_args_restrict]]{
            device_ptr<float> tensor_d(tensor_ptr);
            device_ptr<int8_t> result_d(result_ptr);

            for (int i = 0; i < size; i++){
                result_d[i] = round(tensor_d[i] / scale);
            }
        });
     });

     quant_event.wait();
    //  print_exec_time(quant_event, "Quant");
     free(tensor_ptr, q);

     return result_ptr;
}

// // Dequantise the input TENSOR with the given SCALE.
// float* dequant(queue &q, int size, float scale, int8_t *tensor_ptr)
// {
//     float* result_ptr = (float*) malloc_device(size * sizeof(float), q);

//     auto dequant_event = q.submit([&](handler &h) {
//         h.single_task<class DequantID>([=]() [[intel::kernel_args_restrict]]{
//             device_ptr<int8_t> tensor_d(tensor_ptr);
//             device_ptr<float> result_d(result_ptr);
            
//             for (int i = 0; i < size; i++){
//                 result_d[i] = tensor_d[i] * scale;
//             }
//         });
//     });
//     dequant_event.wait();
//     print_exec_time(dequant_event, "Dequant");
//     free(tensor_ptr, q);

//     return result_ptr;
// }

// /* Carry out MaxPool on the given TENSOR (C * H * W) with a stride of 2. */
int8_t* max_pool_q(queue &q, int chn, int row, int col, int8_t *tensor_ptr)
{
    const int stride = 2;
    const int nr = 1 + (row - 1) / stride;
    const int nc = 1 + (col - 1) / stride;

    int8_t* result_ptr = (int8_t*) malloc_device(chn * nr * nc * sizeof(int8_t), q);

    auto pool_event = q.submit([&](handler &h) {
    
        h.single_task<class MaxpoolID>([=]() [[intel::kernel_args_restrict]]{
            device_ptr<int8_t> tensor_d(tensor_ptr);
            device_ptr<int8_t> result_d(result_ptr);

            for (int i = 0; i < chn * nr * nc; i++){
                int index[3] = {i / (nr * nc), (i / nc) % nr, i % nc};
                int max_r = (index[1] + 1) * stride;
                int max_c = (index[2] + 1) * stride;
                int8_t f = numeric_limits<int8_t>::lowest();
                for (int i = index[1] * stride; i < max_r; i++) {
                    for (int j = index[2] * stride; j < max_c; j++) {
                        int8_t cur = tensor_d[index[0] * row * col + i * col + j];
                        f = f > cur ? f : cur;
                    }
                }
                result_d[index[0] * nr * nc + index[1] * nc + index[2]] = f;
            }
        });
    });
    pool_event.wait();
    // print_exec_time(pool_event, "Pool");
    free(tensor_ptr, q);

    return result_ptr;
}


int main()
{
    cout.precision(4);

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#elif CPU
    auto selector = cpu_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
    property_list queue_properties{sycl::property::queue::enable_profiling()};
    queue q = sycl::queue(selector, exception_handler, queue_properties);

    auto device = q.get_device();

    // if (!device.has(sycl::aspect::usm_device_allocations)) {
    //   std::cerr << "This design must either target a board that supports USM "
    //                "Host/Shared allocations, or IP Component Authoring. "
    //             << std::endl;
    //   std::terminate();
    // }
    // if (!device.get_info<info::device::usm_host_allocations>()) {
    //   std::cerr << "ERROR: The selected device does not support USM host"
    //             << " allocations\n";
    //   std::terminate();
    // }

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // The file that encodes all parameters of the model.
    ifstream rf_data("/home/u196631/urop/final-year-project/data/model_params_quant.mmzk", ios::binary);
    if (!rf_data.is_open())
    {
        cout << "Cannot open file!" << std::endl;
        return 1;
    }

    cout << "Model parameters file opened successfully" << std::endl;

    /* reading input from file ----- */ 
    std::ifstream inputFile("/home/u196631/urop/final-year-project/src/input.txt");
    std::vector<float> values;
    std::string line, value;
  
  if (inputFile.is_open()) {
        while (std::getline(inputFile, line)) {
                std::istringstream iss(line);
                while (std::getline(iss, value, ',')) {
                float floatValue = std::stof(value);
                values.push_back(floatValue);
        }
  }
  inputFile.close();

  // Convert vector to float array
  float* input_ff = new float[values.size()];
  std::copy(values.begin(), values.end(), input_ff);
  
  /* input from file stored in input_ff ----- */ 

    // Get parameters
    int8_t *weights1 = read_param_int8(rf_data);
    int32_t *biases1 = read_param_int32(rf_data);
    int8_t *weights2 = read_param_int8(rf_data);
    int32_t *biases2 = read_param_int32(rf_data);
    float *prototypes = read_param_float(rf_data);
    int8_t *fc_weights = read_param_int8(rf_data);

    rf_data.close();

    float *logits_f = new float[3];
    float *upsampled_f = new float[15 * 224 * 224];

    // Timings.
    long times[N] = {};

    for (int i = 0; i < N; i++)
    {     
        auto start = high_resolution_clock::now();        
        
        // Allocate device memory for input image
      float* input_f_ptr = (float*) malloc_device(3 * 224 * 224 * sizeof(float), q);
      auto input_to_device_event = q.memcpy(input_f_ptr, input_ff, 3*224*224 * sizeof(float));
      input_to_device_event.wait();

        // Quantise the input.
        int8_t* input_ptr = quant(q, 3 * 224 * 224, 0.01979798823595047, input_f_ptr);
        // Convolutional layers (* 2).
        // input 3 * 224 * 224, output 64*224*224
        int8_t* conved1_ptr = conv_pad_q<Conv1ID, 3, 224, 64>(q, input_ptr, weights1, biases1, 0.01979798823595047, 0.013484773226082325, 0.04881289601325989);

        // input 64*224*224, output 64*112*122
        int8_t* pooled1_ptr = max_pool_q(q, 64, 224, 224, conved1_ptr);

        // input 64*112*122, output  512*112*112
        // int8_t* conved2_ptr = conv_pad_q<Conv2ID, 64, 112, 512>(q, pooled1_ptr, weights2, biases2, 0.04881289601325989, 0.0006907337228767574, 0.016132580116391182);
        
    //     // // input 512*112*112, output 512*56*56/
    //     // int8_t* pooled2_ptr = max_pool_q(q, 512, 112, 112, conved2_ptr);
    //     // float *pooled2_f_ptr = dequant(q, 512 * 56 * 56, 0.016132580116391182, pooled2_ptr);

        q.wait();
        
        auto stop = high_resolution_clock::now();
        times[i] = duration_cast<microseconds>(stop - start).count();
    }

    delete[] input_ff;


    // // Output timings.
    // long total = 0;
    // for (int i = 0; i < N; i++)
    // {
    //     total += times[i];
    //     cout << "Iteration " << (i + 1) << ": " << (float)times[i] / 1e6 << std::endl;
    // }

    // cout << "Total: " << (float)total / 1e6 << std::endl;

    delete[] weights1;
    delete[] biases1;
    delete[] weights2;
    delete[] biases2;
    delete[] prototypes;
    delete[] fc_weights;

    delete[] logits_f;
    delete[] upsampled_f;
  } else {
    std::cout << "Failed to open the input file." << std::endl;
    return 1;
  }
              
}