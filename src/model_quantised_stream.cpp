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

#include "aiaibl.hpp"
#include "params.hpp"
#include "input_handler.hpp"
#include "utils.hpp"
#include "constants.hpp"

using namespace std::chrono;
using namespace std;
using namespace sycl;

// Number of iterations.
#define N 1

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

    // Enable the queue profiling to time the execution
    property_list queue_properties{sycl::property::queue::enable_profiling()};
    queue q = sycl::queue(selector, exception_handler, queue_properties);

    auto device = q.get_device();

    if (!device.has(sycl::aspect::usm_device_allocations)) {
      std::cerr << "This design must either target a board that supports USM "
                   "Host/Shared allocations, or IP Component Authoring. "
                << std::endl;
      std::terminate();
    }
    if (!device.get_info<info::device::usm_host_allocations>()) {
        std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations\n";
        std::terminate();
    }

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // The file that encodes all parameters of the model.
    ifstream rf_data("/home/u178815/final-year-project/data/model_params_quant.mmzk", ios::binary);
    if (!rf_data.is_open())
    {
        cout << "Cannot open file!" << std::endl;
        return 1;
    }

    cout << "Model parameters file opened successfully" << std::endl;

    // Get parameters
    int8_t *weights1 = read_param_int8(rf_data);
    int32_t *biases1 = read_param_int32(rf_data);
    int8_t *weights2 = read_param_int8(rf_data);
    int32_t *biases2 = read_param_int32(rf_data);
    float *prototypes = read_param_float(rf_data);
    int8_t *fc_weights = read_param_int8(rf_data);

    rf_data.close();
    cout << "Model parameters read" << std::endl;

    std::vector<float> logits;
    std::vector<float> upsample;

    logits.resize(num_classes);
    upsample.resize(num_protos*img_dim*img_dim);

    using QuantToConvPipe = pipe<
        class QuantToConvPipeID,
        std::array<int8_t,img_pad_chn_size>,
        img_chn>;
    
    using Conv1ToPool1Pipe = pipe<
        class Conv11ToPool1PipeID,
        std::array<int8_t,img_chn_size>,
        c1_chn>;
    
    using Pool1ToConv2Pipe = pipe<
        class Pool1ToConv2PipeID,
        std::array<int8_t,c1_pad_chn_size>,
        c1_chn>;
    
    using Conv2ToPool2Pipe = pipe<
        class Conv2ToPool2PipeID,
        std::array<int8_t,c1_chn_size>,
        c2_chn>;
    
    using Pool2ToSimMapPipe = pipe<
        class Pool2ToSimMapPipeID,
        std::array<int8_t,c2_chn_size>,
        c2_chn>;
    
    using SimMapToScorePipe = pipe<
        class SimMapToScorePipeID,
        std::array<int8_t,c2_chn_size>,
        num_protos>;
    
    using SimMapToScorePipe = pipe<
        class SimMapToScorePipeID,
        std::array<int8_t,c2_chn_size>,
        num_protos>;
    
    
    
    
    
    
    


    cout << "Init outputs" << std::endl;

    /*
    Allocate memory and copy onto device for the network parameters and layers 
    */
    float* img_f_ptr = malloc_shared<float>(img_size, q);
    // int8_t* img_ptr = (int8_t*) malloc_device(img_pad_size * sizeof(int8_t), q);
    // for (int i = 0; i < img_pad_size; i++) img_ptr[i] = 0;
    
    int8_t* conv1_f_ptr = (int8_t*) malloc_device(c1_chn * img_chn * filter_size * sizeof(int8_t), q);
    int32_t* conv1_b_ptr = (int32_t*) malloc_device(img_chn * sizeof(int32_t), q);
    // int8_t* conved1_ptr = (int8_t*) malloc_device(c1_chn * img_dim * img_dim * sizeof(int8_t), q);
    auto filter1_to_device_event = q.memcpy(conv1_f_ptr, &weights1[0], c1_chn * img_chn * filter_size * sizeof(int8_t));
    auto bias1_to_device_event = q.memcpy(conv1_b_ptr, &biases1[0], img_chn * sizeof(int32_t));

    // int8_t* pooled1_ptr = (int8_t*) malloc_device(c1_chn * c1_pad_dim * c1_pad_dim * sizeof(int8_t), q);
    // for (int i = 0; i < c1_chn*c1_pad_dim*c1_pad_dim; i++) pooled1_ptr[i] = 0;

    int8_t* conv2_f_ptr = (int8_t*) malloc_device(c2_chn * c1_chn * filter_size * sizeof(int8_t), q);
    int32_t* conv2_b_ptr = (int32_t*) malloc_device(c1_chn * sizeof(int32_t), q);
    // int8_t* conved2_ptr = (int8_t*) malloc_device(c2_chn * c1_dim * c1_dim * sizeof(int8_t), q);
    auto filter2_to_device_event = q.memcpy(conv2_f_ptr, &weights2[0], c2_chn * c1_chn * filter_size * sizeof(int8_t));
    auto bias2_to_device_event = q.memcpy(conv2_b_ptr, &biases2[0], c1_chn * sizeof(int32_t));

    // int8_t* pooled2_ptr = (int8_t*) malloc_device(c2_chn * c2_dim * c2_dim * sizeof(int8_t), q);

    // float* fm_ptr = (float*) malloc_device(c2_chn * c2_dim * c2_dim * sizeof(float), q);

    // float* sims_ptr = (float*) malloc_device(num_protos*c2_dim*c2_dim*sizeof(float), q);
    float* proto_ptr = (float*) malloc_device(num_protos*c2_chn*sizeof(float), q);
    auto proto_to_device_event = q.memcpy(proto_ptr, &prototypes[0], num_protos*c2_chn*sizeof(float));

    // int8_t* sim_score_ptr = (int8_t*) malloc_device(num_protos*sizeof(int8_t), q);

    int8_t* weights_ptr = (int8_t*) malloc_device(num_classes * num_protos * sizeof(int8_t), q);
    float* logits_ptr = (float*) malloc_shared(num_classes*sizeof(float), q);
    auto weights_to_device_event = q.memcpy(weights_ptr, &fc_weights[0], num_classes * num_protos * sizeof(int8_t));

    float* upsample_ptr = (float*) malloc_device(num_protos*img_dim*img_dim*sizeof(float), q);

    cout << "Pointers initialised\n";

    // Timings.
    double times[N] = {};

    for (int i = 0; i < N; i++)
    {
        // Copy input image into device (TODO: stream from host instead)
        /*
            Quantise the image
        */
        auto quant_event = q.submit([&](handler &h) {
            h.single_task<class Quantise>([=]() [[intel::kernel_args_restrict]]{
                host_ptr<float> img_d(img_f_ptr);
                device_ptr<int8_t> result_d(img_ptr);

                for(int c = 0; c < img_chn; c++){
                    for (int y = 0; y < img_dim; y++){
                        [[intel::ivdep]]
                        for (int x = 0; x < img_dim; x++){
                            result_d[c*img_pad_chn_size + (1+y)*img_pad_dim + x+1] = round(img_d[c*img_chn_size+ y*img_dim + x] / 0.01979798823595047);
                        }
                    }
                }
            });
        });
        
        /*
            Convolute image with 3x3xc1_chn filter and zero-padding, quantise and ReLU output features
        */

        auto c1_corner_event = q.submit([&](handler &h)
        {
            h.depends_on(quant_event);
            h.depends_on(filter1_to_device_event);
            h.depends_on(bias1_to_device_event);
            
            h.single_task<class Conv1>([=]() [[intel::kernel_args_restrict]]{
                Conv<img_chn,img_pad_dim,c1_chn,filter_size>(img_ptr, conv1_f_ptr, conv1_b_ptr, c1_scale, conved1_ptr);
            });
        });

        /*
            Max Pooling with stride 2 on feature map
        */
        auto pool1_event = q.submit([&](handler &h) {
            h.depends_on(c1_corner_event);

            h.single_task<class Maxpool1>([=]() [[intel::kernel_args_restrict]]{
                device_ptr<int8_t> in_d(conved1_ptr);
                device_ptr<int8_t> result_d(pooled1_ptr);

                constexpr int out_dim = img_dim / 2;
                for (int c = 0; c < c1_chn; c++){
                    constexpr int _c = (out_dim+2)*(out_dim+2);
                    for (int y = 0; y < out_dim; y++){
                        #pragma unroll 2
                        [[intel::ivdep]]
                        for (int x = 0; x < out_dim; x++){
                            int8_t res = sycl::max(in_d[c*img_dim*img_dim + y*img_dim + x], in_d[c*img_dim*img_dim + y*img_dim + x]);
                            res = sycl::max(res, in_d[c*img_dim*img_dim + (y+1)*img_dim + x]);
                            res = sycl::max(res, in_d[c*img_dim*img_dim + (y+1)*img_dim + x + 1]);
                            result_d[_c*c + (y+1)*(out_dim+2) + x+1] = res;
                        }
                    }
                }
            });
        });

        /*
            Convolute image with 3x3xc2_chn filter and zero-padding, quantise and ReLU output features
        */
        auto c2_corner_event = q.submit([&](handler &h)
        {
            h.depends_on(pool1_event);
            h.depends_on(filter2_to_device_event);
            h.depends_on(bias2_to_device_event);
            
            h.single_task<class Conv2>([=]() [[intel::kernel_args_restrict]]{
                Conv<c1_chn,c1_pad_dim,c2_chn,filter_size>(pooled1_ptr, conv2_f_ptr, conv2_b_ptr, c2_scale, conved2_ptr);
            });
        });

        /*
            Max Pooling with stride 2 on feature map
        */
        auto pool2_event = q.submit([&](handler &h) {
            h.depends_on(c2_corner_event);
            
            h.single_task<class Maxpool2>([=]() [[intel::kernel_args_restrict]]{
                MaxPool<c2_chn, c1_dim>(conved2_ptr, pooled2_ptr);
            });
        });

        /*
            Dequantise the feature map
        */
        auto dequant_event = q.submit([&](handler &h) {
            h.depends_on(pool2_event);
            h.single_task<class Dequantise>([=]() [[intel::kernel_args_restrict]]{
                device_ptr<int8_t> in_d(pooled2_ptr);
                device_ptr<float> result_d(fm_ptr);

                [[intel::ivdep]]
                for (int i = 0; i < c2_chn*c2_dim*c2_dim; i++){
                    result_d[i] = in_d[i] * 0.016132580116391182;
                }
            });
        });

        /*
            Calculate similarity between feature map values and prototypes
        */
        auto similarity_event = q.submit([&](handler &h) {
            h.depends_on(proto_to_device_event);
            h.depends_on(dequant_event);

            h.single_task<class SimMap>([=]() [[intel::kernel_args_restrict]]{
                #pragma clang fp contract(fast)
                device_ptr<float> in_d(fm_ptr);
                device_ptr<float> proto_d(proto_ptr);
                device_ptr<float> res_d(sims_ptr);

                for (int p = 0; p < num_protos; p++){
                    // Load proto values into private memory
                    [[intel::fpga_register]]
                    float cur_proto[c2_chn];
                    #pragma unroll
                    for (int i = 0; i < c2_chn; i++){
                        cur_proto[i] = proto_d[p*c2_chn + i];
                    }
                    for (int y = 0; y < c2_dim; y++){
                        for (int x = 0; x < c2_dim; x++){
                            const int _y = y*c2_dim + x;
                            float res = 0.0f;
                            #pragma unroll 2
                            [[intel::ivdep]]
                            for (int c = 0; c < c2_chn; c++){
                                float dist = in_d[c*c2_chn_size + _y] - cur_proto[c];
                                res += dist*dist;
                            }
                            res_d[p*c2_dim*c2_dim + y*c2_dim + x] = sycl::log((res+1)/(res+0.0001f));
                        }
                    }
                }
            });

        });

        /*
            Linear 2D Upsample by a scale factor 4 without alligning the corners 
        */
        auto upsample1_event = q.submit([&](handler &h) {
            h.depends_on(similarity_event);

            h.single_task<class Upsample1>([=]() [[intel::kernel_args_restrict]]{
                device_ptr<float> tensor_d(sims_ptr);
                device_ptr<float> result_d(upsample_ptr);
                for (int index = 0; index < num_protos; index++){
                    // auto _q = (index[1] + 2) / 4;
                    for (int i = 0; i < 2; i++) {
                        [[intel::ivdep]]
                        for (int j = 0; j < 2; j++){
                            result_d[(index*c2_dim*c2_dim) + (i * c2_dim) + j] = tensor_d[index * c2_dim * c2_dim];
                            result_d[(index * c2_dim * c2_dim) + (i * c2_dim) + 4 * c2_dim - 1 - j] = tensor_d[(index * c2_dim * c2_dim) + c2_dim - 1];
                            result_d[(index * c2_dim * c2_dim) + (4 * c2_dim - 1 - i) * c2_dim + j] = tensor_d[(index * c2_dim * c2_dim) + (c2_dim - 1) * c2_dim];
                            result_d[(index * c2_dim * c2_dim) + (4 * c2_dim - 1 - i) * c2_dim + 4 * c2_dim - 1 - j] = tensor_d[(index * c2_dim * c2_dim) + ((c2_dim - 1) * c2_dim) + c2_dim - 1];
                        }
                    }
                }
            });

        });

        auto upsample2_event = q.submit([&](handler &h) {
            h.depends_on(similarity_event);

            h.single_task<class Upsample2>([=]() [[intel::kernel_args_restrict]]{
                device_ptr<float> tensor_d(sims_ptr);
                device_ptr<float> result_d(upsample_ptr);
                for (int c = 0; c < num_protos; c++){
                    for (int index = 0; index < (c2_dim * 4 - 4); index++){
                        auto _r = 2* (index % 4) + 1;
                        auto _q = index / 4;
                        for (int i = 0; i < 2; i++){
                            result_d[(c * c2_dim * c2_dim) + (i * c2_dim) + index + 2] = (tensor_d[(c * c2_dim * c2_dim) + _q] * (8 - _r) + tensor_d[(c * c2_dim * c2_dim) + _q + 1] * _r) / 8;
                            result_d[(c * c2_dim * c2_dim) + (4 * c2_dim - 1 - i) * c2_dim + index + 2] = (tensor_d[(c * c2_dim * c2_dim) + (c2_dim - 1) * c2_dim + _q] * (8 - _r) + tensor_d[(c * c2_dim * c2_dim) + (c2_dim - 1) * c2_dim + _q + 1] * _r) / 8;
                        }     
                    }
                }
                
            });
        });

        auto upsample3_event = q.submit([&](handler &h) {
            h.single_task<class Upsample3>([=] () [[intel::kernel_args_restrict]]{
                device_ptr<float> tensor_d(sims_ptr);
                device_ptr<float> result_d(upsample_ptr);
                for (int c = 0; c < num_protos; c++){
                    for (int index = 0; index < (c2_dim * 4 - 4); index++){
                        auto _r = 2 * (c % 4) + 1;
                        auto _q = c / 4;
                        for (int i = 0; i < 2; i++) {
                            result_d[(index * c2_dim * c2_dim) + (c + 2) * c2_dim + i] = (tensor_d[(index * c2_dim * c2_dim) + _q * c2_dim] * (8 - _r) + tensor_d[(index * c2_dim * c2_dim) + (_q + 1) * c2_dim] * _r) / 8;
                            result_d[(index * c2_dim * c2_dim) + (c + 2) * c2_dim + 4 * c2_dim - 1 - i] = (tensor_d[(index * c2_dim * c2_dim) + _q * c2_dim + c2_dim - 1] * (8 - _r) + tensor_d[(index * c2_dim * c2_dim) + (_q + 1) * c2_dim + c2_dim - 1] * _r) / 8;
                        }
                    }
                }
            });
        });

        auto upsample4_event = q.submit([&](handler &h) {
            h.single_task<class Upsample4>([=] () [[intel::kernel_args_restrict]]{
                device_ptr<float> tensor_d(sims_ptr);
                device_ptr<float> result_d(upsample_ptr);
                for (int c = 0; c < num_protos; c++){
                    for (int y = 0; y < c2_dim * 4 - 4; y++){
                        for (int x = 0; x < c2_dim * 4 - 4; x++){
                            auto _r1 = 2 * (y % 4) + 1;
                            auto _q1 = y / 4;
                            auto _r2 = 2 * (x % 4) + 1;
                            auto _q2 = x / 4;
                            result_d[(c * c2_dim * c2_dim) + (y + 2) * c2_dim + x + 2]
                                = (tensor_d[(c * c2_dim * c2_dim) + _q1 * c2_dim +_q2] * (8 - _r1) + tensor_d[(c * c2_dim * c2_dim) + (_q1 + 1) * c2_dim + _q2] * _r1) * (8 - _r2) / 64
                                + (tensor_d[(c * c2_dim * c2_dim) + _q1 * c2_dim + _q2 + 1] * (8 - _r1) + tensor_d[(c * c2_dim * c2_dim) + (_q1 + 1) * c2_dim + _q2 + 1] * _r1) * _r2 / 64;
                        }
                    }
                }
            });
        });

        /*
            Calculate similarity score between feature map and prototypes using top-9 averaging
        */
        auto sim_score_event = q.submit([&](handler &h) {
            h.depends_on(similarity_event);

            h.single_task<class SimScore>([=]() [[intel::kernel_args_restrict]]{
                device_ptr<float> in_d(sims_ptr);
                device_ptr<int8_t> result_d(sim_score_ptr);

                for (int c = 0; c < num_protos; c++){
                    float r_[9];
                    for (int i = 0; i < 9; i++) {
                        r_[i] = in_d[c * c2_dim*c2_dim + i];
                        int k = i;
                        while (true) {
                            if ((r_[k] >= r_[k / 2]) ) break;
                            float temp = r_[k];
                            r_[k] = r_[k / 2];
                            r_[k / 2] = temp;
                            k /= 2;
                        }
                    }
                
                    for (int i = 9; i < c2_dim*c2_dim; i++) {
                        if (in_d[c * c2_dim*c2_dim + i] > r_[0]) {
                            r_[0] = in_d[c * c2_dim*c2_dim + i];
                            int k = 0;
                            while (k < 9) {
                                if (k >= 4) break;

                                if (r_[k] > r_[2 * k + 1] || r_[k] > r_[2 * k + 2]) {
                                    float temp = r_[k];
                                    if (r_[2 * k + 1] < r_[2 * k + 2]) {
                                        r_[k] = r_[2 * k + 1];
                                        r_[2 * k + 1] = temp;
                                        k = 2 * k + 1;
                                    } else {
                                        r_[k] = r_[2 * k + 2];
                                        r_[2 * k + 2] = temp;
                                        k = 2 * k + 2;
                                    }
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                    float res = (r_[0] + r_[1] + r_[2] + r_[3] + r_[4] + r_[5] + r_[6] + r_[7] + r_[8]) / 9;
                    result_d[c] = round(res / 0.01979798823595047);
                }
            });
        });

        /*
            Fully Connected layer between prototypes and classification score
        */
        auto fc_event = q.submit([&](handler &h) {
            h.depends_on(sim_score_event);
            h.depends_on(weights_to_device_event);

            h.single_task<class FC>([=]() [[intel::kernel_args_restrict]]{
                device_ptr<int8_t> in_d(sim_score_ptr);
                device_ptr<int8_t> w_d(weights_ptr);
                device_ptr<float> result_d(logits_ptr);

                for (int index = 0; index < num_classes; index++){
                    // The scales are hardcoded for the sole fully-connected layer of our model.
                    int32_t sum = 0;
                    for (int i = 0; i < num_protos; i++) {
                        sum += in_d[i] * w_d[index * num_protos + i];
                    }
                    result_d[index] = sum * 0.01979798823595047*0.009601877070963383;
                }
            });
        });

        cout << "Jobs submitted \n";
        // Wait for final kenels to finish execution
        fc_event.wait();
        upsample1_event.wait();
        upsample2_event.wait();
        upsample3_event.wait();
        upsample4_event.wait();

        times[i] += print_exec_time(quant_event, "Quant");
        times[i] += print_exec_time(c1_corner_event, "C1 Corner");
        times[i] += print_exec_time(pool1_event, "Pool1");
        times[i] += print_exec_time(c2_corner_event, "C2 Corner");
        times[i] += print_exec_time(pool2_event, "Pool2");
        times[i] += print_exec_time(dequant_event, "Dequant");
        times[i] += print_exec_time(similarity_event, "Similarity");
        times[i] += print_exec_time(upsample1_event, "Upsample 1");
        times[i] += print_exec_time(upsample2_event, "Upsample 2");
        times[i] += print_exec_time(upsample3_event, "Upsample 3");
        times[i] += print_exec_time(upsample4_event, "Upsample 4");
        times[i] += print_exec_time(sim_score_event, "Sim Score");
        times[i] += print_exec_time(fc_event, "FC Event");
    }

    auto logits_to_host = q.submit([&] (handler &h) {
        h.memcpy(logits.data(), logits_ptr, num_classes*sizeof(float));
    });
    logits_to_host.wait();

    auto upsample_to_host = q.submit([&] (handler &h) {
        h.memcpy(upsample.data(), upsample_ptr, num_protos*img_dim*img_dim*sizeof(float));
    });
    upsample_to_host.wait();

    free(img_f_ptr, q);
    free(img_ptr, q);
    free(conv1_f_ptr, q);
    free(conv1_b_ptr, q);
    free(conved1_ptr, q);
    free(pooled1_ptr, q);
    free(conv2_f_ptr, q);
    free(conv2_b_ptr, q);
    free(conved2_ptr, q);
    free(pooled2_ptr, q);
    free(fm_ptr, q);
    free(sims_ptr, q);
    free(proto_ptr, q);
    free(sim_score_ptr, q);
    free(weights_ptr, q);
    free(logits_ptr, q);

    //Print out the output.
    peek(1, 3, &logits[0], true); // The index corresponding to the maximum value is the index of the chosen classification. 0 For cabbage; 1 for carrot; 2 for tomato.
    // // peek(1, 15, avg_f, false);
    peek(224, 224, &upsample[0], true);
    // peek(224, 224, upsampled_f + 224 * 224, true);
    // peek(224, 224, upsampled_f + 14 * 224 * 224, true);

    // Output timings.
    long total = 0;
    for (int i = 0; i < N; i++)
    {
        total += times[i];
        cout << "Iteration " << (i + 1) << ": " << (float)times[i] / 1e6 << std::endl;
    }

    cout << "Total: " << (float)total / 1e6 << std::endl;

    delete[] weights1;
    delete[] biases1;
    delete[] weights2;
    delete[] biases2;
    delete[] prototypes;
    delete[] fc_weights;
    return 0;
}