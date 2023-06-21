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
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include "compute_units.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"
#include "aiaibl.hpp"
#include "utils.hpp"

using namespace std::chrono;
using namespace std;
using namespace sycl;

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif
#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }

template <std::size_t ID> class Quant1ComputeUnits;
template <std::size_t ID> class DequantComputeUnits;
template <std::size_t ID> class Conv1FilterComputeNodes;
template <std::size_t ID,std::size_t PID> class Conv2FilterComputeNodes;
template <std::size_t ID> class Conv1AccComputeNodes;
template <std::size_t ID> class Conv2AccComputeNodes;
template <std::size_t ID> class MaxPool1ComputeNodes;
template <std::size_t ID> class MaxPool2ComputeNodes;

// Number of iterations.
#define N 1

void print_exec_time(event e, string name){
    auto start_time = e.template
            get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time = e.template
            get_profiling_info<sycl::info::event_profiling::command_end>();
    double dur = (end_time - start_time) / 1.0e9;

    cout <<  name << " event time: " << dur << std::endl;
}
/* Show the matrix. If snapshot = true, only shwo the first 5 * 5 corner. */
void peek(int row, int col, float *matrix, bool snapshot)
{
    int peek_level = 16;
    for (int i = 0; i < (snapshot && row > peek_level ? peek_level : row); i++)
    {
        for (int j = 0; j < (snapshot && col > peek_level ? peek_level : col); j++)
        {
            cout << std::fixed << matrix[i * col + j] << "\t";
        }
        if (col > peek_level && snapshot)
        {
            cout << "...";
        }
        cout << std::endl;
    }
    if (row > peek_level && snapshot)
    {
        cout << "...";
    }
    cout << std::endl;
}

/* Show the matrix. If snapshot = true, only shwo the first 5 * 5 corner. */
void peek_int(int row, int col, ac_int<8, true> *matrix, bool snapshot)
{
    for (int i = 0; i < (snapshot && row > 5 ? 5 : row); i++)
    {
        for (int j = 0; j < (snapshot && col > 5 ? 5 : col); j++)
        {
            cout << std::fixed << (int32_t)(matrix[i * col + j]) << "\t";
        }
        if (col > 5 && snapshot)
        {
            cout << "...";
        }
        cout << std::endl;
    }
    if (row > 5 && snapshot)
    {
        cout << "...";
    }
    cout << std::endl;
}

/* Carry out MaxPool on the given TENSOR (C * H * W) with a stride of 2. */
ac_int<8, true>* max_pool_q(queue &q, int chn, int row, int col, ac_int<8, true> *tensor_ptr)
{
    const int stride = 2;
    const int nr = 1 + (row - 1) / stride;
    const int nc = 1 + (col - 1) / stride;

    ac_int<8, true>* result_ptr = (ac_int<8, true>*) malloc_device(chn * nr * nc * sizeof(ac_int<8, true>), q);

    auto pool_event = q.submit([&](handler &h) {
    
        h.single_task<class MaxpoolID>([=]() [[intel::kernel_args_restrict]]{
            device_ptr<ac_int<8, true>> tensor_d(tensor_ptr);
            device_ptr<ac_int<8, true>> result_d(result_ptr);

            for (int i = 0; i < chn * nr * nc; i++){
                int index[3] = {i / (nr * nc), (i / nc) % nr, i % nc};
                int max_r = (index[1] + 1) * stride;
                int max_c = (index[2] + 1) * stride;
                ac_int<8, true> f = numeric_limits<ac_int<8, true>>::lowest();
                for (int i = index[1] * stride; i < max_r; i++) {
                    for (int j = index[2] * stride; j < max_c; j++) {
                        ac_int<8, true> cur = tensor_d[index[0] * row * col + i * col + j];
                        f = f > cur ? f : cur;
                    }
                }
                result_d[index[0] * nr * nc + index[1] * nc + index[2]] = f;
            }
        });
    });
    pool_event.wait();
    print_exec_time(pool_event, "Pool");
    free(tensor_ptr, q);

    return result_ptr;
}

// Dequantise the input TENSOR with the given SCALE.
float* dequant(queue &q, int size, float scale, ac_int<8, true> *tensor_ptr)
{
    float* result_ptr = (float*) malloc_device(size * sizeof(float), q);

    auto dequant_event = q.submit([&](handler &h) {
        h.single_task<class DequantID>([=]() [[intel::kernel_args_restrict]]{
            device_ptr<ac_int<8, true>> tensor_d(tensor_ptr);
            device_ptr<float> result_d(result_ptr);
            
            for (int i = 0; i < size; i++){
                result_d[i] = tensor_d[i] * scale;
            }
        });
    });
    dequant_event.wait();
    print_exec_time(dequant_event, "Dequant");
    free(tensor_ptr, q);

    return result_ptr;
}

/* Carry out quantised convolution (with padding) between the TENSOR (1 * CHN * W * W) and the FILTER (D * CHN * 3 * 3), then ReLU. */
ac_int<8, true>* conv_pad_q(queue &q, int chn, int size, ac_int<8, true> *tensor_ptr, ac_int<8, true> *filter, int d, int32_t *biases, float tensor_scale, float filter_scale, float result_scale)
{
    ac_int<8, true>* filter_ptr = (ac_int<8, true>*) malloc_device(d * chn * 9 * sizeof(ac_int<8, true>), q);
    int32_t* bias_ptr = (int32_t*) malloc_device(d * sizeof(int32_t), q);
    ac_int<8, true>* result_ptr = (ac_int<8, true>*) malloc_device(d * size * size * sizeof(ac_int<8, true>), q);

    auto filter_to_device_event = q.memcpy(filter_ptr, &filter[0], d * chn * 9 * sizeof(ac_int<8, true>));
    auto bias_to_device_event = q.memcpy(bias_ptr, &biases[0], d * sizeof(int32_t));

    auto corner_event = q.submit([&](handler &h)
    {
        h.depends_on(filter_to_device_event);
        h.depends_on(bias_to_device_event);
        
        h.single_task<class ConvCornerID>([=]() [[intel::kernel_args_restrict]]{
            device_ptr<ac_int<8, true>> tensor_d(tensor_ptr);
            device_ptr<ac_int<8, true>> filter_d(filter_ptr);
            device_ptr<int32_t> bias_d(bias_ptr);
            device_ptr<ac_int<8, true>> result_d(result_ptr);

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

        h.single_task<class ConvBoundID>([=]() {
            device_ptr<ac_int<8, true>> tensor_d(tensor_ptr);
            device_ptr<ac_int<8, true>> filter_d(filter_ptr);
            device_ptr<int32_t> bias_d(bias_ptr);
            device_ptr<ac_int<8, true>> result_d(result_ptr);

            const float scale = tensor_scale * filter_scale / result_scale;
            for (int i = 0; i < d * (size - 2); i++) {
                int index[2] = { i / (size - 2), i % (size - 2) };
                int32_t sum = 0;
                for (int c = 0; c < chn; c++) {
                    int _fi = index[0] * chn + c;
                    #pragma unroll
                    for (int i = 0; i <= 1; i++) {
                        #pragma unroll
                        for (int j = 0; j <= 2; j++) {
                            sum += filter_d[_fi * 9 + (i+1) * 3 + j] * tensor_d[c * size * size + (i * size) + index[1] + j];
                        }
                    }
                }

                sum += bias_d[index[0]];
                result_d[index[0] * size * size + index[1] + 1] = sum > 0 ? round(sum * scale) : 0;

                sum = 0;
                for (int c = 0; c < chn; c++) {
                    int _fi = index[0] * chn + c;
                    #pragma unroll
                    for (int i = -2; i <= -1; i++) {
                        #pragma unroll
                        for (int j = 0; j <= 2; j++) {
                            sum += filter_d[_fi * 9 + (i + 2) * 3 + j] * tensor_d[c * size * size + (size + i) * size + index[1] + j];      
                        }
                    }
                }

                sum += bias_d[index[0]];
                result_d[index[0] * size * size + (size - 1) * size + index[1] + 1] = sum > 0 ? round(sum * scale) : 0;

                sum = 0;
                for (int c = 0; c < chn; c++) {
                    int _fi = index[0] * chn + c;
                    #pragma unroll
                    for (int i = 0; i <= 2; i++) {
                        #pragma unroll
                        for (int j = 0; j <= 1; j++) {
                            sum += filter_d[_fi * 9 + i * 3 + j + 1] * tensor_d[c * size * size + (index[1] + i) * size + j];        
                        }
                    }
                }

                sum += bias_d[index[0]];
                result_d[index[0] * size * size + (index[1] + 1) * size] = sum > 0 ? round(sum * scale) : 0;

                sum = 0;
                for (int c = 0; c < chn; c++) {
                    int _fi = index[0] * chn + c;
                    #pragma unroll
                    for (int i = 0; i <= 2; i++) {
                        #pragma unroll
                        for (int j = -2; j <= -1; j++) {
                        sum += filter_d[_fi * 9 + i * 3 + j + 2] * tensor_d[c * size * size + (index[1] + i) * size + size + j];        
                        }
                    }
                }

                sum += bias_d[index[0]];
                result_d[index[0] * size * size + (index[1] + 1) * size + size - 1] = sum > 0 ? round(sum * scale) : 0;
            }
        });
    });

    // Task for interior elements (that uses all 3 * 3 filters).
    auto interior_event = q.submit([&](handler &h) {
        h.depends_on(filter_to_device_event);
        h.depends_on(bias_to_device_event);

        h.single_task<class ConvIntrID>([=]() {
            device_ptr<ac_int<8, true>> tensor_d(tensor_ptr);
            device_ptr<ac_int<8, true>> filter_d(filter_ptr);
            device_ptr<int32_t> bias_d(bias_ptr);
            device_ptr<ac_int<8, true>> result_d(result_ptr);

            const float scale = tensor_scale * filter_scale / result_scale;
            for (int i = 0; i < (size - 2) * (size - 2); i++){
                int index[3] = { i / ((size - 2) * (size - 2)), (i / (size -2)) % (size - 2), i % (size - 2) };
                int32_t sum = 0;
                #pragma unroll 2 // Partial unrolling for the outermost loop.
                for (int c = 0; c < chn; c++) {
                    int _fi = index[0] * chn + c;
                    #pragma unroll
                    for (int i = 0; i <= 2; i++) {
                        #pragma unroll
                        for (int j = 0; j <= 2; j++) {
                            sum += filter_d[_fi * 9 + i * 3 + j] * tensor_d[c * size * size + (index[1] + i) * size + index[2] + j];        
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

    print_exec_time(corner_event, "Corner conv");
    print_exec_time(boundry_event, "Boundry conv");
    print_exec_time(interior_event, "Interior conv");

    free(tensor_ptr, q);
    free(filter_ptr, q);
    free(bias_ptr, q);

    return result_ptr;
}

/* Carry out the calculation for a fully-connected layer. */
ac_int<8, true>* fully_connected(queue &q, int c_in, int c_out, ac_int<8, true> *tensor_ptr,
                     ac_int<8, true> *weights)
{
    ac_int<8, true>* weights_ptr = (ac_int<8, true>*) malloc_device(c_out * c_in * sizeof(ac_int<8, true>), q);
    ac_int<8, true>* result_ptr = (ac_int<8, true>*) malloc_device(c_out * sizeof(ac_int<8, true>), q);

    auto weights_to_device_event = q.memcpy(weights_ptr, &weights[0], c_out * c_in * sizeof(ac_int<8, true>));

    auto fc_event = q.submit([&](handler &h) {
        h.depends_on(weights_to_device_event);

        h.single_task<class FC>([=] () [[intel::kernel_args_restrict]]{
            device_ptr<ac_int<8, true>> tensor_d(tensor_ptr);
            device_ptr<ac_int<8, true>> weights_d(weights_ptr);
            device_ptr<ac_int<8, true>> result_d(result_ptr);

            for (int index = 0; index < c_out; index++){
                // The scales are hardcoded for the sole fully-connected layer of our model.
                const float tensor_scale = 0.01979798823595047;
                const float filter_scale = 0.009601877070963383;
                const float result_scale = 0.06617073714733123779;
                const float scale = tensor_scale * filter_scale / result_scale;
                int32_t sum = 0;
                for (int i = 0; i < c_in; i++) {
                    sum += tensor_d[i] * weights_d[index * c_in + i];
                }
                result_d[index] = round(sum * scale);
            }
        });
    });

    fc_event.wait();
    print_exec_time(fc_event, "FC");

    free(weights_ptr, q);
    free(tensor_ptr, q);

    return result_ptr;
}

/* The L2-distance computation, used for the prototype layer. */
float* l2_distance(queue &q, int chn, int length, float *tensor_ptr, int p_len, float *prototypes)
{
    float* result_ptr = (float*) malloc_device(p_len * length * sizeof(float), q);
    float* proto_ptr = (float*) malloc_device(p_len * chn * sizeof(float), q);

    auto dist_event = q.submit([&](handler &h) {
        h.single_task<class L2Dist>([=] () [[intel::kernel_args_restrict]]{
                #pragma clang fp contract(fast)
            device_ptr<float> tensor_d(tensor_ptr);
            device_ptr<float> proto_d(proto_ptr);
            device_ptr<float> result_d(result_ptr);

            for (int proto_idx = 0; proto_idx < p_len; proto_idx++){
                for (int feat_idx = 0; feat_idx < length; feat_idx++){
                    float sum = 0.0f;
                    int p = proto_idx * chn;
                    for (int c = 0; c < chn; c++) {

                        float dist = tensor_d[c * length + feat_idx] - proto_d[p + c];
                        sum += dist*dist;
                    }
                    result_d[proto_idx * length + feat_idx] = sycl::sqrt(sum);
                }
            }
        });
    });

    dist_event.wait();
    print_exec_time(dist_event, "Distance");
    free(tensor_ptr, q);
    free(proto_ptr, q);

    return result_ptr;
}

/* Convert distances to similarity map (part of the prototype layer). */
float* distance_2_similarity(queue &q, int length, float *tensor_ptr)
{
    float* result_ptr = (float*) malloc_device(length * sizeof(float), q);
    auto sim_event = q.submit([&](handler &h) {
        h.single_task<class Dist2Sim>([=] () [[intel::kernel_args_restrict]]{
            device_ptr<float> tensor_d(tensor_ptr);
            device_ptr<float> result_d(result_ptr);

            for (int index = 0; index < length; index++){
                result_d[index] = log((tensor_d[index] + 1) / (tensor_d[index] + 0.0001f));
            }
        });
    });
    sim_event.wait();
    print_exec_time(sim_event, "Dist 2 Sim");
    free(tensor_ptr, q);

    return result_ptr;
}

/* Pooling that takes the largest (or smallest, based on IS_TOP) 9 elements, then take the average. */
float* top9_average_pooling(queue &q, int chn, int length, float *tensor_ptr)
{
    float* result_ptr = (float*) malloc_device(chn * sizeof(float), q);
    auto top9_event = q.submit([&](handler &h) {
    // The implementation uses a max-heap to keep track of the 9 largest elements.
        h.single_task<class Top9>([=] () [[intel::kernel_args_restrict]]{
            device_ptr<float> tensor_d(tensor_ptr);
            device_ptr<float> result_d(result_ptr);

            for (int index = 0; index < chn; index++){
                float r_[9];
                for (int i = 0; i < 9; i++) {
                    r_[i] = tensor_d[index * length + i];
                    int k = i;
                    while (true) {
                        if ((r_[k] >= r_[k / 2]) ) break;
                        float temp = r_[k];
                        r_[k] = r_[k / 2];
                        r_[k / 2] = temp;
                        k /= 2;
                    }
                }

                for (int i = 9; i < length; i++) {
                    if (tensor_d[index * length + i] > r_[0]) {
                        r_[0] = tensor_d[index * length + i];
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

                result_d[index] = (r_[0] + r_[1] + r_[2] + r_[3] + r_[4] + r_[5] + r_[6] + r_[7] + r_[8]) / 9;
            }
        });
    });
    top9_event.wait();
    print_exec_time(top9_event, "Top 9");

    return result_ptr;
}

/* Pooling that takes the largest (or smallest, based on IS_TOP) 9 elements, then take the average. */
void bottom9_average_pooling(queue &q, int chn, int length, float *tensor, float *result)
{
    {
        buffer m_buf(tensor, range(chn, length));
        buffer r_buf(result, range(chn));
        q.submit([&](handler &h)
                 {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h);

      // The implementation uses a max-heap to keep track of the 9 largest elements.
      h.single_task<class Bottom9>([=] () {
        for (int index = 0; index < chn; index++){
            float r_[9];
            for (int i = 0; i < 9; i++) {
            r_[i] = m[index][i];
            int k = i;
            while (true) {
                if (r_[k] <= r_[k / 2]) break;
                float temp = r_[k];
                r_[k] = r_[k / 2];
                r_[k / 2] = temp;
                k /= 2;
            }
            }

            for (int i = 9; i < length; i++) {
            if (m[index][i] <= r_[0]) {
                r_[0] = m[index][i];
                int k = 0;
                while (k < 9) {
                if (k >= 4) break;

                if (r_[k] <= r_[2 * k + 1] || r_[k] <= r_[2 * k + 2]) {
                    float temp = r_[k];
                    if (r_[2 * k + 1] >= r_[2 * k + 2]) {
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

            r[index] = (r_[0] + r_[1] + r_[2] + r_[3] + r_[4] + r_[5] + r_[6] + r_[7] + r_[8]) / 9;
        }
      });
    });
    }
}

/* Upsample the TENSOR by a factor of 2 using Linear 2D without aligning the corners. */
void upsample4(queue &q, int chn, int row, int col, float *tensor_ptr, float* result)
{
    float* result_ptr = (float*) malloc_device(chn * row * col * sizeof(float) * 16, q);

    auto upsample1_event = q.submit([&](handler &h) {
        h.single_task<class Upsample1>([=] () [[intel::kernel_args_restrict]]{
            device_ptr<float> tensor_d(tensor_ptr);
            device_ptr<float> result_d(result_ptr);
            for (int index = 0; index < chn; index++){
                // auto _q = (index[1] + 2) / 4;
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++){
                        result_d[(index * row * col) + (i * col) + j] = tensor_d[index * row * col];
                        result_d[(index * row * col) + (i * col) + 4 * col - 1 - j] = tensor_d[(index * row * col) + col - 1];
                        result_d[(index * row * col) + (4 * row - 1 - i) * col + j] = tensor_d[(index * row * col) + (row - 1) * col];
                        result_d[(index * row * col) + (4 * row - 1 - i) * col + 4 * col - 1 - j] = tensor_d[(index * row * col) + ((row - 1) * col) + col - 1];
                    }
                }
            }
        });
    });

    auto upsample2_event = q.submit([&](handler &h) {
        h.single_task<class Upsample2>([=] () [[intel::kernel_args_restrict]]{
            device_ptr<float> tensor_d(tensor_ptr);
            device_ptr<float> result_d(result_ptr);
            for (int i = 0; i < chn * (col * 4 - 4); i++){
                int index[2] = {i / (col * 4 -4), i % (col * 4 -4)};
                auto _r = 2 * (index[1] % 4) + 1;
                auto _q = index[1] / 4;
                for (int i = 0; i < 2; i++) {
                result_d[(index[0] * row * col) + (i * col) + index[1] + 2] = (tensor_d[(index[0] * row * col) + _q] * (8 - _r) + tensor_d[(index[0] * row * col) + _q + 1] * _r) / 8;
                result_d[(index[0] * row * col) + (4 * row - 1 - i) * col + index[1] + 2] = (tensor_d[(index[0] * row * col) + (row - 1) * col + _q] * (8 - _r) + tensor_d[(index[0] * row * col) + (row - 1) * col + _q + 1] * _r) / 8;
                }
            }
        });
    });

    auto upsample3_event = q.submit([&](handler &h) {
        h.single_task<class Upsample3>([=] () [[intel::kernel_args_restrict]]{
            device_ptr<float> tensor_d(tensor_ptr);
            device_ptr<float> result_d(result_ptr);
            for (int i = 0; i < chn * (row * 4 - 4); i++){
                int index[2] = {i / (row * 4 -4), i % (row * 4 -4)};
                auto _r = 2 * (index[1] % 4) + 1;
                auto _q = index[1] / 4;
                for (int i = 0; i < 2; i++) {
                    result_d[(index[0] * row * col) + (index[1] + 2) * col + i] = (tensor_d[(index[0] * row * col) + _q * col] * (8 - _r) + tensor_d[(index[0] * row * col) + (_q + 1) * col] * _r) / 8;
                    result_d[(index[0] * row * col) + (index[1] + 2) * col + 4 * col - 1 - i] = (tensor_d[(index[0] * row * col) + _q * col + col - 1] * (8 - _r) + tensor_d[(index[0] * row * col) + (_q + 1) * col + col - 1] * _r) / 8;
                }
            }
        });
    });

    auto upsample4_event = q.submit([&](handler &h) {
        h.single_task<class Upsample4>([=] () [[intel::kernel_args_restrict]]{
            device_ptr<float> tensor_d(tensor_ptr);
            device_ptr<float> result_d(result_ptr);
            for (int i = 0; i < chn * (row * 4 - 4) * (col * 4 - 4); i++){
                int index[3] = {i / ((row * 4 - 4) * (col * 4 - 4)), (i / (col * 4 - 4)) % (row * 4 - 4), i % (col * 4 - 4)};
                auto _r1 = 2 * (index[1] % 4) + 1;
                auto _q1 = index[1] / 4;
                auto _r2 = 2 * (index[2] % 4) + 1;
                auto _q2 = index[2] / 4;
                result_d[(index[0] * row * col) + (index[1] + 2) * col + index[2] + 2]
                    = (tensor_d[(index[0] * row * col) + _q1 * col +_q2] * (8 - _r1) + tensor_d[(index[0] * row * col) + (_q1 + 1) * col + _q2] * _r1) * (8 - _r2) / 64
                    + (tensor_d[(index[0] * row * col) + _q1 * col + _q2 + 1] * (8 - _r1) + tensor_d[(index[0] * row * col) + (_q1 + 1) * col + _q2 + 1] * _r1) * _r2 / 64;
            }
        });
    });

    // write back after all computation
    auto device_to_host = q.submit([&] (handler &h) {
        h.depends_on(upsample1_event);
        h.depends_on(upsample2_event);
        h.depends_on(upsample3_event);
        h.depends_on(upsample4_event);
        
        h.memcpy(&result[0], result_ptr, chn * row * col * sizeof(float) * 16);
    });
    device_to_host.wait();
    print_exec_time(upsample1_event, "Upsample 1");
    print_exec_time(upsample2_event, "Upsample 2");
    print_exec_time(upsample3_event, "Upsample 3");
    print_exec_time(upsample4_event, "Upsample 4");
    free(tensor_ptr, q);
}

// Quantise the input TENSOR with the given SCALE.
ac_int<8, true>* quant(queue &q, int size, float scale, float *tensor_ptr)
{
    ac_int<8, true>* result_ptr = (ac_int<8, true>*) malloc_shared(size * sizeof(ac_int<8, true>), q);

    auto quant_event = q.submit([&](handler &h) {
        h.single_task<class QuantID>([=]() [[intel::kernel_args_restrict]]{
            device_ptr<float> tensor_d(tensor_ptr);
            device_ptr<ac_int<8, true>> result_d(result_ptr);

            for (int i = 0; i < size; i++){
                result_d[i] = round(tensor_d[i] / scale);
            }
        });
     });

     quant_event.wait();
     print_exec_time(quant_event, "Quant");
     free(tensor_ptr, q);

     return result_ptr;
}

template <int dim, int in_burst_s, int out_burst_s>
void conv_top_row(std::array<ac_int<8, true>, in_burst_s> in_burst, ac_int<8, true> *filter, std::array<int32_t, out_burst_s> out_burst){

    out_burst[0] = in_burst[0]*filter[4] /
        + in_burst[1]*filter[5] \
        + in_burst[dim]*filter[7] \
        + in_burst[dim+1]*filter[8];

    for (int i = 0; i < dim-2; i++){

        out_burst[i] = in_burst[i]*filter[3]/
            + in_burst[i+1]*filter[4] \
            + in_burst[i+2]*filter[5] \
            + in_burst[dim+i]*filter[6] \
            + in_burst[dim+i+1]*filter[7] \
            + in_burst[dim+i+2]*filter[8];
    }

    out_burst[dim-1] = in_burst[dim-2]*filter[3] \
            + in_burst[dim-1]*filter[4] \
            + in_burst[2*dim-2]*filter[6] \
            + in_burst[2*dim-1]*filter[7];
}

template <int dim, int in_burst_s, int out_burst_s>
void conv_middle_rows(std::array<ac_int<8, true>, in_burst_s> in_burst, std::array<ac_int<8, true>, in_burst_s> last_burst, ac_int<8, true> *filter, std::array<int32_t, out_burst_s> out_burst){
    out_burst[0] = in_burst[0]*filter[1] \
            + last_burst[1]*filter[2]\
            + last_burst[dim]*filter[4]\
            + last_burst[dim+1]*filter[5]\
            + in_burst[0]*filter[7]\
            + in_burst[1]*filter[8];
    [[intel::ivdep]]
    for (int intr = 0; intr < dim-2; intr++){
        auto sum = 0;
        #pragma unroll
        for (int j = 0; j < 3; j++){
            sum += last_burst[intr+j]*filter[j];
            sum += last_burst[dim+intr+j]*filter[3+j];
            sum += in_burst[intr+j]*filter[6+j];
        }
        out_burst[intr+1] = sum;
    }
    out_burst[dim-1] = last_burst[dim-2]*filter[0] \
        + last_burst[dim-1]*filter[1] \
        + last_burst[2*dim-2]*filter[3] \
        + last_burst[2*dim-1]*filter[4]\
        + in_burst[dim-2]*filter[6]\
        + in_burst[dim-1]*filter[7];
}

template <int dim, int in_burst_s, int out_burst_s>
void conv_bottom_row(std::array<ac_int<8, true>, in_burst_s> in_burst, ac_int<8, true> *filter, std::array<int32_t, out_burst_s> out_burst){
    out_burst[0] = in_burst[0]*filter[1] /
                + in_burst[1]*filter[2] \
                + in_burst[dim]*filter[4] \
                + in_burst[dim+1]*filter[5];
            [[intel_ivdep]]
            for (int i = 0; i < dim-2; i++){

                out_burst[i] = in_burst[i]*filter[0]/
                    + in_burst[i+1]*filter[1] \
                    + in_burst[i+2]*filter[2] \
                    + in_burst[dim+i]*filter[3] \
                    + in_burst[dim+i+1]*filter[4] \
                    + in_burst[dim+i+2]*filter[5];
            }

            out_burst[dim-1] = in_burst[dim-2]*filter[0] \
                    + in_burst[dim-1]*filter[1] \
                    + in_burst[2*dim-2]*filter[3] \
                    + in_burst[2*dim-1]*filter[4];
}
int main()
{
    cout.precision(4);

    // Device Setup ##################################################################
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

    // #####################################################################################

    // Load paramters onto host ############################################################
    ifstream rf_data("/home/u178815/final-year-project/data/model_params_quant.mmzk", ios::binary);
    if (!rf_data.is_open())
    {
        cout << "Cannot open file!" << std::endl;
        return 1;
    }

    cout << "Model parameters file opened successfully" << std::endl;

    // Get parameters
    ac_int<8, true> *weights1 = read_param_int8(rf_data);
    int32_t *biases1 = read_param_int32(rf_data);
    ac_int<8, true> *weights2 = read_param_int8(rf_data);
    int32_t *biases2 = read_param_int32(rf_data);
    float *prototypes = read_param_float(rf_data);
    ac_int<8, true> *fc_weights = read_param_int8(rf_data);

    rf_data.close();
    cout << "Model parameters read" << std::endl;

    // Initialise contant values for the model #############################################
    std::vector<float> logits_f;
    std::vector<float> upsampled_f;

    logits_f.resize(num_classes);
    upsampled_f.resize(num_protos*img_dim*img_dim);

    // Pipes between the layers of the network ############################################
    using Quant1Pipes = fpga_tools::PipeArray<
                            class Quant1ToConv1PipesID,
                            std::array<ac_int<8, true>, img_burst_size>,
                            img_bursts,
                            img_chn, c1_chn>;
    using Conv1FilterPipes = fpga_tools::PipeArray<
                            class Conv1FilterToAccPipesID,
                            std::array<int32_t, conv1_burst_size>,
                            conv1_bursts,
                            img_chn, c1_chn>;
    using Conv1AccPipes = fpga_tools::PipeArray<
                            class Conv1AccToPoolPipesID,
                            std::array<ac_int<8, true>, conv1_burst_size>,
                            conv1_bursts,
                            c1_chn>;
    using Pool1Pipes = fpga_tools::PipeArray<
                            class PoolToConv2PipesID,
                            std::array<ac_int<8, true>, pool1_burst_size>,
                            pool1_bursts,
                            c1_chn, c2_chn>;
    using Conv2FilterPipes = fpga_tools::PipeArray<
                            class Conv2FilterToAccPipesID,
                            std::array<int32_t, conv2_burst_size>,
                            conv2_bursts,
                            c1_chn, c2_chn>;
    using Conv2AccPipes = fpga_tools::PipeArray<
                            class Conv2AccToPoolPipesID,
                            std::array<ac_int<8, true>, conv2_burst_size>,
                            conv2_bursts,
                            c2_chn>;
    using Pool2Pipes = fpga_tools::PipeArray<
                            class PoolToDequantPipesID,
                            std::array<ac_int<8, true>, pool2_burst_size>,
                            pool2_bursts,
                            c2_chn>;
    using DequantPipes = fpga_tools::PipeArray<
                            class PoolToDequantPipesID,
                            std::array<float, dequant_burst_size>,
                            dequant_bursts,
                            c2_chn, num_protos>;

    // Load image and parameters into device ###############################################
    float *img_f_ptr[img_chn];
    for (int i = 0; i < img_chn; i++){
        if ((img_f_ptr[i] = malloc_device<float>(img_size, q)) == nullptr) {
                    std::cerr << "ERROR: could not allocate space for 'img_f_ptr'\n";
                    std::terminate();
            }
        q.memcpy(img_f_ptr[i], &input_f[i*img_size], img_size * sizeof(float)).wait();
    }

    ac_int<8, true>* conv1_f_ptr = (ac_int<8, true>*) malloc_device(c1_chn * img_chn * filter_size * sizeof(ac_int<8, true>), q);
    int32_t* conv1_b_ptr = (int32_t*) malloc_device(img_chn * sizeof(int32_t), q);
    ac_int<8, true>* conv2_f_ptr = (ac_int<8, true>*) malloc_device(c2_chn * c1_chn * filter_size * sizeof(ac_int<8, true>), q);
    int32_t* conv2_b_ptr = (int32_t*) malloc_device(c1_chn * sizeof(int32_t), q);
    

    q.memcpy(conv1_f_ptr, &weights1[0], c1_chn * img_chn * filter_size * sizeof(ac_int<8, true>)).wait();
    q.memcpy(conv1_b_ptr, &biases1[0], img_chn * sizeof(int32_t)).wait();
    q.memcpy(conv2_f_ptr, &weights2[0], c2_chn * c1_chn * filter_size * sizeof(ac_int<8, true>)).wait();
    q.memcpy(conv2_b_ptr, &biases2[0], c1_chn * sizeof(int32_t)).wait();

    cout << "Initialised image" << std::endl;

    // Timings.
    long times[N] = {};

    for (int i = 0; i < N; i++)
    {
        float *input_f = load_image(i);
        auto start = high_resolution_clock::now();

        // Enqueue QUANTISATION kernals for each channel ################################################
        SubmitComputeUnits<img_chn, Quant1ComputeUnits>(q, [=](auto ID) [[intel::kernel_args_restrict]]{
            device_ptr<float> img_d(img_f_ptr[ID]);
            for (int burst_idx = 0; burst_idx < img_dim / 2; burst_idx++ ){
                std::array<ac_int<8, true>, img_burst_size> ddr_read;
                #pragma unroll
                for (int k = 0; k < img_burst_size; k++){
                    ddr_read[k] = round(img_d[burst_idx * img_burst_size + k] / img_scale);
                }
                fpga_tools::UnrolledLoop<c1_chn>([&ID, ddr_read](auto PID){
                    Quant1Pipes::PipeAt<ID,PID>::write(ddr_read);
                });
            }
        });
        cout << "Pipelined to Conv1" << std::endl;

        // ac_int<8, true> *input = (ac_int<8, true>*) malloc_shared(3 * 448 * 448 * sizeof(ac_int<8, true>), q);
        
        // float* input_f_ptr = (float*) malloc_device(3 * 224 * 224 * sizeof(float), q);
        // q.memcpy(input_f_ptr, &input_f[0], 3*224*224 * sizeof(float)).wait();
        // ac_int<8, true>* input_old = quant(q, 3 * 224 * 224, 0.01979798823595047, input_f_ptr);

        // Enqueue CONVFILTER kernals for each channel ################################################
        SubmitComputeUnits<c1_chn, Conv1FilterComputeNodes>(q, [=] (auto ID) [[intel::kernel_args_restrict]]{
            device_ptr<ac_int<8, true>> filter_d(conv1_f_ptr);
            constexpr int in_chn = ID / c1_chn;
            constexpr int out_chn = ID % c1_chn;
            ac_int<8, true> *filter = &filter_d[out_chn*img_chn*filter_size + in_chn*filter_size];
            
            std::array<ac_int<8, true>, img_burst_size> in_burst = Quant1Pipes::PipeAt<in_chn, out_chn>::read();
            std::array<int32_t, conv1_burst_size> out_burst;

            conv_top_row<img_dim, img_burst_size, conv1_burst_size>(in_burst, filter, out_burst);
            Conv1FilterPipes::PipeAt<in_chn, out_chn>::write(out_burst);

            std::array<ac_int<8, true>, img_burst_size> last_burst = in_burst;
            for (int i = 0; i < (img_dim / 2) - 1; i++){
                in_burst = Quant1Pipes::PipeAt<in_chn, out_chn>::read();
                std::array<int32_t, conv1_burst_size> out_burst;

                conv_middle_rows<img_dim, img_burst_size, conv1_burst_size>(in_burst, last_burst, filter, out_burst);

                Conv1FilterPipes::PipeAt<in_chn, out_chn>::write(out_burst);
                last_burst = in_burst;
                
            }
            conv_bottom_row<img_dim, img_burst_size, conv1_burst_size>(in_burst, filter, out_burst);

            Conv1FilterPipes::PipeAt<in_chn, out_chn>::write(out_burst);
            // write to unique pipes (img_dim x con1_dim of them) where next kernel reads from img_dim pipes

            // for (int i =0; i < img_dim / 2; i++){
            //     std::array<ac_int<8, true>, img_burst_size> img_burst = Quant1Pipes::PipeAt<ID,0>::read();
            //     #pragma unroll
            //     for (int k =0; k < img_burst_size; k++){
            //         input[ID*img_size + i*img_burst_size+k] = img_burst[k];
            //     }
            // }
        });
        cout << "Pipelined to ConvAcc" << std::endl;
        // q.wait();
        // for (int i =0; i < img_size*img_chn; i++){
        //     if (input[i] != input_old[i]){
        //         cout << input[i] << " vs " << input_old << " old\n";
        //     }
        // }

        // Enqueue CONVACC kernals for each channel ################################################
        SubmitComputeUnits<c1_chn, Conv1AccComputeNodes>(q, [=] (auto ID) [[intel::kernel_args_restrict]]{
            device_ptr<int32_t> bias_d(conv1_b_ptr);
            for (int i = 0; i < img_dim; i++){
                int32_t *bursts[img_chn];
                fpga_tools::UnrolledLoop<img_chn>([&ID, &bursts](auto CID){
                    bursts[CID] = &Conv1FilterPipes::PipeAt<CID, ID>::read()[0];
                });
                std::array<ac_int<8, true>, conv1_burst_size> out_burst;
                #pragma unroll
                for (int cur =0; cur < img_dim; cur++){
                    int32_t sum = 0;
                    #pragma unroll
                    for (int c = 0 ; c < img_chn; c++){
                        sum += bursts[c][cur];
                    }
                    sum += bias_d[ID];
                    out_burst[cur] = sum > 0 ? round(sum * scale) : 0;
                }
                Conv1AccPipes::PipeAt<ID>::write(out_burst);
            }
        });
        cout << "Pipelined to MaxPool" << std::endl;

        // Enqueue MAXPOOl kernals for each channel ################################################
        SubmitComputeUnits<c1_chn, MaxPool1ComputeNodes>(q, [=] (auto ID) [[intel::kernel_args_restrict]]{
            for (int i = 0; i < c1_dim / 2; i++){
                std::array<ac_int<8, true>, pool1_burst_size> out_burst;
                for (int k = 0; k < 2; k++){
                    std::array<ac_int<8, true>, conv1_burst_size> in_burst = Conv1AccPipes::PipeAt<ID>::read();
                    for (int j = 0; j < c1_dim; j++){
                        out_burst[k*c1_dim+j] = sycl::max(in_burst[j], in_burst[j+1]);
                    }
                    in_burst = Conv1AccPipes::PipeAt<ID>::read();
                    for (int j = 0; j < c1_dim; j++){
                        out_burst[k*c1_dim+j] = sycl::max(out_burst[k*c1_dim+j],sycl::max(in_burst[j], in_burst[j+1]));
                    }
                }
                fpga_tools::UnrolledLoop<c2_chn>([&ID, out_burst](auto PID){
                    Pool1Pipes::PipeAt<ID,PID>::write(out_burst);
                });
            }
        });

        // Enqueue CONVFILTER kernals for each channel ################################################
        fpga_tools::UnrolledLoop<c1_chn>([&q, &conv2_f_ptr](auto in_chn){
            SubmitComputeUnits_N<c2_chn, Conv2FilterComputeNodes, in_chn>(q, [=] (auto ID) [[intel::kernel_args_restrict]]{
                device_ptr<ac_int<8, true>> filter_d(conv2_f_ptr);
                constexpr int out_chn = ID;
                ac_int<8, true> *filter = &filter_d[out_chn*c1_chn*filter_size + in_chn*filter_size];
                
                std::array<ac_int<8, true>, pool1_burst_size> in_burst = Pool1Pipes::PipeAt<in_chn, out_chn>::read();
                std::array<int32_t, conv2_burst_size> out_burst;

                conv_top_row<c1_dim, pool1_burst_size, conv2_burst_size>(in_burst, filter, out_burst);
                Conv2FilterPipes::PipeAt<in_chn, out_chn>::write(out_burst);

                std::array<ac_int<8, true>, pool1_burst_size> last_burst = in_burst;
                for (int i = 0; i < (c1_dim / 2) - 1; i++){
                    in_burst = Pool1Pipes::PipeAt<in_chn, out_chn>::read();
                    std::array<int32_t, conv2_burst_size> out_burst;

                    conv_middle_rows<c1_dim, pool1_burst_size, conv2_burst_size>(in_burst, last_burst, filter, out_burst);

                    Conv2FilterPipes::PipeAt<in_chn, out_chn>::write(out_burst);
                    last_burst = in_burst;
                    
                }
                conv_bottom_row<c1_dim, pool1_burst_size, conv2_burst_size>(in_burst, filter, out_burst);

                Conv2FilterPipes::PipeAt<in_chn, out_chn>::write(out_burst);
            });
        });

        // Enqueue CONVACC kernals for each channel ################################################
        SubmitComputeUnits<c2_chn, Conv2AccComputeNodes>(q, [=] (auto ID) [[intel::kernel_args_restrict]]{
            device_ptr<int32_t> bias_d(conv2_b_ptr);
            for (int i = 0; i < c1_dim; i++){
                int32_t *bursts[c1_chn];
                fpga_tools::UnrolledLoop<c1_chn>([&ID, &bursts](auto CID){
                    bursts[CID] = &Conv2FilterPipes::PipeAt<CID, ID>::read()[0];
                });
                std::array<ac_int<8, true>, conv2_burst_size> out_burst;
                #pragma unroll
                for (int cur =0; cur < c2_dim; cur++){
                    int32_t sum = 0;
                    #pragma unroll
                    for (int c = 0 ; c < c2_chn; c++){
                        sum += bursts[c][cur];
                    }
                    sum += bias_d[ID];
                    out_burst[cur] = sum > 0 ? round(sum * scale) : 0;
                }
                Conv2AccPipes::PipeAt<ID>::write(out_burst);
            }
        });

        // Enqueue MAXPOOl kernals for each channel ################################################
        SubmitComputeUnits<c2_chn, MaxPool2ComputeNodes>(q, [=] (auto ID) [[intel::kernel_args_restrict]]{
            for (int i = 0; i < c2_dim; i++){
                std::array<ac_int<8, true>, pool2_burst_size> out_burst;

                std::array<ac_int<8, true>, conv2_burst_size> in_burst = Conv2AccPipes::PipeAt<ID>::read();
                for (int j = 0; j < c2_dim; j++){
                    out_burst[c2_dim+j] = sycl::max(in_burst[j], in_burst[j+1]);
                }
                in_burst = Conv2AccPipes::PipeAt<ID>::read();
                for (int j = 0; j < c2_dim; j++){
                    out_burst[c2_dim+j] = sycl::max(out_burst[c2_dim+j],sycl::max(in_burst[j], in_burst[j+1]));
                }
            
                Pool2Pipes::PipeAt<ID>::write(out_burst);
            }
        });

        // Enqueue DEQUANTISATION kernals for each channel ################################################
        SubmitComputeUnits<c2_chn, DequantComputeUnits>(q, [=](auto ID) [[intel::kernel_args_restrict]]{
            for (int burst_idx = 0; burst_idx < c2_dim; burst_idx++ ){
                std::array<ac_int<8, true>, pool2_burst_size> in_burst;
                std::array<float, dequant_burst_size> out_burst;
                
                #pragma unroll
                for (int k = 0; k < dequant_burst_size; k++){
                    out_burst[k] = in_burst[burst_idx * pool2_burst_size + k] * img_scale;
                }
                fpga_tools::UnrolledLoop<num_protos>([&ID, out_burst](auto PID){
                    DequantPipes::PipeAt<ID, PID>::write(out_burst);
                });
            }
        });

        // Enqueue Distance^2 kernals for each channel ################################################
        fpga_tools::UnrolledLoop<c1_chn>([&q, &conv2_f_ptr](auto in_chn){
        SubmitComputeUnits_N<c2_chn, Conv2FilterComputeNodes, in_chn>(q, [=] (auto ID) [[intel::kernel_args_restrict]]{
        SubmitComputeUnits<c2_chn, DequantComputeUnits>(q, [=](auto ID) [[intel::kernel_args_restrict]]{
            for (int burst_idx = 0; burst_idx < c2_dim; burst_idx++ ){
                std::array<ac_int<8, true>, pool2_burst_size> in_burst;
                std::array<float, dequant_burst_size> out_burst;
                
                #pragma unroll
                for (int k = 0; k < dequant_burst_size; k++){
                    out_burst[k] = in_burst[burst_idx * pool2_burst_size + k] * img_scale;
                }
                fpga_tools::UnrolledLoop<num_protos>([&ID, out_burst](auto PID){
                    DequantPipes::PipeAt<ID, PID>::write(out_burst);
                });
            }
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
                device_ptr<ac_int<8, true>> result_d(sim_score_ptr);

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
                device_ptr<ac_int<8, true>> in_d(sim_score_ptr);
                device_ptr<ac_int<8, true>> w_d(weights_ptr);
                device_ptr<float> result_d(logits_ptr);

                for (int index = 0; index < num_classes; index++){
                    // The scales are hardcoded for the sole fully-connected layer of our model.
                    int32_t sum = 0;
                    for (int i = 0; i < num_protos; i++) {
                        sum += in_d[i] * w_d[index * num_protos + i];
                    }
                    result_d[index] = round(sum * 0.01979798823595047*0.009601877070963383);
                }
            });
        });

        auto stop = high_resolution_clock::now();
        times[i] = duration_cast<microseconds>(stop - start).count();
    }

//     // Print out the output.
//     peek(1, 3, logits_f, true); // The index corresponding to the maximum value is the index of the chosen classification. 0 For cabbage; 1 for carrot; 2 for tomato.
//     // peek(1, 15, avg_f, false);
//     peek(224, 224, upsampled_f, true);
//     peek(224, 224, upsampled_f + 224 * 224, true);
//     peek(224, 224, upsampled_f + 14 * 224 * 224, true);

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