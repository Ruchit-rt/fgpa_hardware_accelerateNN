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
void peek_int(int row, int col, int8_t *matrix, bool snapshot)
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
    print_exec_time(pool_event, "Pool");
    free(tensor_ptr, q);

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
     print_exec_time(quant_event, "Quant");
     free(tensor_ptr, q);

     return result_ptr;
}

// Dequantise the input TENSOR with the given SCALE.
float* dequant(queue &q, int size, float scale, int8_t *tensor_ptr)
{
    float* result_ptr = (float*) malloc_device(size * sizeof(float), q);

    auto dequant_event = q.submit([&](handler &h) {
        h.single_task<class DequantID>([=]() [[intel::kernel_args_restrict]]{
            device_ptr<int8_t> tensor_d(tensor_ptr);
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
template <template <std::size_t ID> typename Name, int chn, int size, int d>
/* Carry out quantised convolution (with padding) between the TENSOR (1 * CHN * W * W) and the FILTER (D * CHN * 3 * 3), then ReLU. */
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

    print_exec_time(corner_event, "Corner conv");
    print_exec_time(boundry_event, "Boundry conv");
    print_exec_time(interior_event, "Interior conv");

    free(tensor_ptr, q);
    free(filter_ptr, q);
    free(bias_ptr, q);

    return result_ptr;
}

/* Carry out the calculation for a fully-connected layer. */
int8_t* fully_connected(queue &q, int c_in, int c_out, int8_t *tensor_ptr,
                     int8_t *weights)
{
    int8_t* weights_ptr = (int8_t*) malloc_device(c_out * c_in * sizeof(int8_t), q);
    int8_t* result_ptr = (int8_t*) malloc_device(c_out * sizeof(int8_t), q);

    auto weights_to_device_event = q.memcpy(weights_ptr, &weights[0], c_out * c_in * sizeof(int8_t));

    auto fc_event = q.submit([&](handler &h) {
        h.depends_on(weights_to_device_event);

        h.single_task<class FC>([=] () [[intel::kernel_args_restrict]]{
            device_ptr<int8_t> tensor_d(tensor_ptr);
            device_ptr<int8_t> weights_d(weights_ptr);
            device_ptr<int8_t> result_d(result_ptr);

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
            device_ptr<float> inp_d(tensor_ptr);
            device_ptr<float> proto_d(proto_ptr);
            device_ptr<float> result_d(result_ptr);

            for (int proto_idx = 0; proto_idx < p_len; proto_idx++){
                for (int feat_idx = 0; feat_idx < length; feat_idx++){
                    float sum = 0.0f;
                    int p = proto_idx * chn;
                    #pragma clang contract(fast)
                    for (int c = 0; c < chn; c++) {

                        float dist = inp_d[c * length + feat_idx] - proto_d[p + c];
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
        
        //  int8_t* conved2_ptr = conv_pad_q<64>(q, 112, pooled1_ptr, weights2, 512, biases2, 0.04881289601325989, 0.0006907337228767574, 0.016132580116391182);
        // input 64*112*122, output  512*112*112
        int8_t* conved2_ptr = conv_pad_q<Conv2ID, 64, 112, 512>(q, pooled1_ptr, weights2, biases2, 0.04881289601325989, 0.0006907337228767574, 0.016132580116391182);
        
        // input 512*112*112, output 512*56*56/
        int8_t* pooled2_ptr = max_pool_q(q, 512, 112, 112, conved2_ptr);
        float *pooled2_f_ptr = dequant(q, 512 * 56 * 56, 0.016132580116391182, pooled2_ptr);

        // Prototype layer.
        float *distances_f_ptr = l2_distance(q, 512, 56 * 56, pooled2_f_ptr, 15, prototypes);
        float *similarities_f_ptr = distance_2_similarity(q, 15 * 56 * 56, distances_f_ptr);
        float *avg_f_ptr = top9_average_pooling(q, 15, 56 * 56, similarities_f_ptr);

        // Compute upsampled activation map (information for interpretation).
        upsample4(q, 15, 56, 56, similarities_f_ptr, upsampled_f);
        
        // Fully-connected layer.
        int8_t *avg_ptr = quant(q, 15, 0.01979798823595047, avg_f_ptr);
        int8_t *logits_ptr = fully_connected(q, 15, 3, avg_ptr, fc_weights);
        float *logits_f_ptr = dequant(q, 3, 0.06617073714733124, logits_ptr);


        // Compute min_distance (information for interpretation).
        // bottom9_average_pooling(q, 15, 56 * 56, distances_f, avg_f);


        q.wait();
        auto logits_to_host = q.submit([&] (handler &h) {
            h.memcpy(&logits_f[0], logits_f_ptr, 3 * sizeof(float));
        });
        logits_to_host.wait();

        auto stop = high_resolution_clock::now();
        times[i] = duration_cast<microseconds>(stop - start).count();
    }
    delete[] input_ff;

    // Print out the output.
    // The index corresponding to the maximum value is the 
    // index of the chosen classification. 0 For cabbage; 1 for carrot; 2 for tomato.
    cout << "Peeking... ";
    peek(1, 3, logits_f, true);
    // peek(1, 15, avg_f, false);
    peek(224, 224, upsampled_f, true);
    peek(224, 224, upsampled_f + 224 * 224, true);
    peek(224, 224, upsampled_f + 14 * 224 * 224, true);

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

    delete[] logits_f;
    delete[] upsampled_f;
    // delete[] input_ff;

    return 0;
    }
    else {
   std::cout << "Failed to open the input file." << std::endl;
   return 1;
  }
}