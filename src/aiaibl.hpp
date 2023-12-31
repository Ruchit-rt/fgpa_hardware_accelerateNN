#ifndef __AIAIBL_HPP__
#define __AIAIBL_HPP__

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


void conv_cond(queue &q, int chn, int row, int col, int8_t *tensor,
    int d, int size, int8_t *filter, int32_t *biases, int8_t *result) {
  {
    buffer m_buf(tensor, range(chn, row, col));
    buffer f_buf(filter, range(d * chn, size, size));
    buffer b_buf(biases, range(d));
    buffer r_buf(result, range(d, row, col));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor f(f_buf, h, read_only);
      accessor b(b_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(d, row, col), [=](auto index) {
        float sum = 0.0f;
        int _d = index[0];
        int _batch = index[0] / d;
        for (int c = 0; c < chn; c++) {
          for (int i = - size / 2; i <= size / 2; i++) {
            for (int j = - size / 2; j <= size / 2; j++) {
              if (index[1] + i < row && index[2] + j < col) {
                sum += f[_d * chn + c][i + size / 2][j + size / 2] * m[_batch * chn + c][index[1] + i][index[2] + j];
              }
            }
          }
        }
        r[index] = sum + b[_d];
      });
    });
  }
}

template <int in_chn,
          int in_dim,
          int out_chn,
          int k_size
          >
void ConvCorners (int8_t *input_ptr, int8_t *filter_ptr, int32_t *bias_ptr, float scale, int8_t* result_ptr) {
    device_ptr<int8_t> input_d(input_ptr);
    device_ptr<int8_t> filter_d(filter_ptr);
    device_ptr<int32_t> bias_d(bias_ptr);
    device_ptr<int8_t> result_d(result_ptr);

    for (int index = 0; index < out_chn; index++){
        int32_t sum = 0;
        for (int c = 0; c < in_chn; c++) {
            int _fi = index * in_chn + c;
            sum += filter_d[_fi * k_size + 4] * input_d[c * in_dim * in_dim];
            sum += filter_d[_fi * k_size + 5] * input_d[c * in_dim * in_dim + 1];
            sum += filter_d[_fi * k_size + 7] * input_d[c * in_dim * in_dim + in_dim];
            sum += filter_d[_fi * k_size + 8] * input_d[c * in_dim * in_dim + in_dim + 1];
        }

        sum += bias_d[index];
        result_d[index * in_dim * in_dim] = sum > 0 ? round(sum * scale) : 0;

        sum = 0;
        for (int c = 0; c < in_chn; c++) {
            int _fi = index * in_chn + c;
            sum += filter_d[_fi * 9 + 3] * input_d[c * in_dim * in_dim + in_dim - 2];
            sum += filter_d[_fi * 9 + 4] * input_d[c * in_dim * in_dim + in_dim - 1];
            sum += filter_d[_fi * 9 + 6] * input_d[c * in_dim * in_dim + in_dim + in_dim - 2];
            sum += filter_d[_fi * 9 + 7] * input_d[c * in_dim * in_dim + in_dim + in_dim - 1];
        }

        sum += bias_d[index];
        result_d[index * in_dim *in_dim + in_dim - 1] = sum > 0 ? round(sum * scale) : 0;

        sum = 0;
        for (int c = 0; c < in_chn; c++) {
            int _fi = index * in_chn + c;
            sum += filter_d[_fi * 9 + 1] * input_d[c * in_dim * in_dim + (in_dim - 2) * in_dim];        
            sum += filter_d[_fi * 9 + 2] * input_d[c * in_dim * in_dim + (in_dim - 2) * in_dim + 1];        
            sum += filter_d[_fi * 9 + 4] * input_d[c * in_dim * in_dim + (in_dim - 1) * in_dim];        
            sum += filter_d[_fi * 9 + 5] * input_d[c * in_dim * in_dim + (in_dim - 1) * in_dim + 1];        
        }

        sum += bias_d[index];
        result_d[index * in_dim * in_dim + (in_dim - 1) * in_dim] = sum > 0 ? round(sum * scale) : 0;
        
        sum = 0;
        for (int c = 0; c < in_chn; c++) {
            int _fi = index * in_chn + c;
            sum += filter_d[_fi * 9] * input_d[c * in_dim * in_dim + (in_dim - 2) * in_dim + in_dim - 2];
            sum += filter_d[_fi * 9 + 1] * input_d[c * in_dim * in_dim + (in_dim - 2) * in_dim + in_dim - 1];
            sum += filter_d[_fi * 9 + 3] * input_d[c * in_dim * in_dim + (in_dim - 1) * in_dim + in_dim - 2];
            sum += filter_d[_fi * 9 + 4] * input_d[c * in_dim * in_dim + (in_dim - 1) * in_dim + in_dim - 1];
        }

        sum += bias_d[index];
        result_d[index * in_dim * in_dim + (in_dim - 1) * in_dim + in_dim - 1] = sum > 0 ? round(sum * scale) : 0;
    }
}
template <int in_chn,
          int in_dim,
          int out_chn,
          int k_size
          >
void ConvBoundry (int8_t *input_ptr, int8_t *filter_ptr, int32_t *bias_ptr, float scale, int8_t* result_ptr)
{
    device_ptr<int8_t> input_d(input_ptr);
    device_ptr<int8_t> filter_d(filter_ptr);
    device_ptr<int32_t> bias_d(bias_ptr);
    device_ptr<int8_t> result_d(result_ptr);

    for (int i = 0; i < out_chn * (in_dim - 2); i++) {
        int index[2] = { i / (in_dim - 2), i % (in_dim - 2) };
        int32_t sum = 0;
        #pragma unroll 2
        for (int c = 0; c < in_chn; c++) {
            int _fi = index[0] * in_chn + c;
            #pragma unroll
            for (int i = 0; i <= 1; i++) {
                #pragma unroll
                for (int j = 0; j <= 2; j++) {
                    sum += filter_d[_fi * k_size + (i+1) * 3 + j] * input_d[c * in_dim * in_dim + (i * in_dim) + index[1] + j];
                }
            }
        }

        sum += bias_d[index[0]];
        result_d[index[0] * in_dim * in_dim + index[1] + 1] = sum > 0 ? round(sum * scale) : 0;

        sum = 0;
        #pragma unroll 2
        for (int c = 0; c < in_chn; c++) {
            int _fi = index[0] * in_chn + c;
            #pragma unroll
            for (int i = -2; i <= -1; i++) {
                #pragma unroll
                for (int j = 0; j <= 2; j++) {
                    sum += filter_d[_fi * k_size + (i + 2) * 3 + j] * input_d[c * in_dim * in_dim + (in_dim + i) * in_dim + index[1] + j];      
                }
            }
        }

        sum += bias_d[index[0]];
        result_d[index[0] * in_dim * in_dim + (in_dim - 1) * in_dim + index[1] + 1] = sum > 0 ? round(sum * scale) : 0;

        sum = 0;
        #pragma unroll 2
        for (int c = 0; c < in_chn; c++) {
            int _fi = index[0] * in_chn + c;
            #pragma unroll
            for (int i = 0; i <= 2; i++) {
                #pragma unroll
                for (int j = 0; j <= 1; j++) {
                    sum += filter_d[_fi * k_size + i * 3 + j + 1] * input_d[c * in_dim * in_dim + (index[1] + i) * in_dim + j];        
                }
            }
        }

        sum += bias_d[index[0]];
        result_d[index[0] * in_dim * in_dim + (index[1] + 1) * in_dim] = sum > 0 ? round(sum * scale) : 0;

        sum = 0;
        #pragma unroll 2
        for (int c = 0; c < in_chn; c++) {
            int _fi = index[0] * in_chn + c;
            #pragma unroll
            for (int i = 0; i <= 2; i++) {
                #pragma unroll
                for (int j = -2; j <= -1; j++) {
                sum += filter_d[_fi * k_size + i * 3 + j + 2] * input_d[c * in_dim * in_dim + (index[1] + i) * in_dim + in_dim + j];        
                }
            }
        }

        sum += bias_d[index[0]];
        result_d[index[0] * in_dim * in_dim + (index[1] + 1) * in_dim + in_dim - 1] = sum > 0 ? round(sum * scale) : 0;
    }
}

template <int in_chn,
          int in_dim,
          int out_chn,
          int k_size
          >
void ConvInterior (int8_t *input_ptr, int8_t *filter_ptr, int32_t *bias_ptr, float scale, int8_t* result_ptr){
    device_ptr<int8_t> input_d(input_ptr);
    device_ptr<int8_t> filter_d(filter_ptr);
    device_ptr<int32_t> bias_d(bias_ptr);
    device_ptr<int8_t> result_d(result_ptr);

    for (int outc = 0; outc < out_chn; outc++){
        for (int y = 0; y < in_dim-2; y++){
            for (int x = 0; x < in_dim-2; x++){
                int32_t sum = 0;
                for (int c = 0; c < in_chn; c++) {
                    int _fi = outc * in_chn + c;
                    #pragma unroll
                    for (int i = 0; i <= 2; i++) {
                        #pragma unroll
                        for (int j = 0; j <= 2; j++) {
                            sum += filter_d[_fi * k_size + i * 3 + j] * input_d[c * in_dim * in_dim + (y + i) * in_dim + x + j];        
                        }
                    }
                }

                sum += bias_d[outc];
                result_d[outc * in_dim * in_dim + (y + 1) * in_dim + x + 1] = sum > 0 ? round(sum * scale) : 0;
            }
        }
    }

}

template <int in_chn,
          int in_dim,
          int out_chn,
          int k_size
          >
void Conv (int8_t *input_ptr, int8_t *filter_ptr, int32_t *bias_ptr, float scale, int8_t* result_ptr){
    // #pragma clang fp contract(fast)
    device_ptr<int8_t> input_d(input_ptr);
    device_ptr<int8_t> filter_d(filter_ptr);
    device_ptr<int32_t> bias_d(bias_ptr);
    device_ptr<int8_t> result_d(result_ptr);
    constexpr int in_str_dim = in_dim-2;

    for (int outc = 0; outc < out_chn; outc++){
        for (int y = 0; y < in_str_dim; y++){
            [[intel::ivdep]]
            for (int x = 0; x < in_str_dim; x++){
                //offset for cur filter
                [[intel::fpga_register]]
                const int _fi = outc*in_chn*k_size;
                
                // Load filter into private memory
                [[intel::fpga_register]]
                int filter[in_chn*k_size];
                #pragma unroll
                for (int i = 0; i < in_chn*k_size; i++){
                    filter[i] = filter_d[_fi + i];
                }

                [[intel::fpga_register]]
                int32_t sum = 0;
                #pragma unroll 2
                [[intel::ivdep]]
                for (int c = 0; c < in_chn; c++) {
                
                    [[intel::fpga_register]]
                    const int _c = c * in_dim * in_dim;

                    #pragma unroll
                    for (int i = 0; i <= 2; i++) {
                        #pragma unroll
                        for (int j = 0; j <= 2; j++) {
                            sum += filter[i * 3 + j] * input_d[_c + (y + i) * in_dim + x + j];    
                        }
                    }
                }

                sum += bias_d[outc];
                result_d[outc * in_str_dim * in_str_dim + y * in_str_dim + x] = sum > 0 ? round(sum * scale) : 0;
            }
        }
    }
}

template <int in_chn,
          int in_dim,
          int out_chn,
          int k_size
          >
void Conv_reg_shift (int8_t *input_ptr, int8_t *filter_ptr, int32_t *bias_ptr, float scale, int8_t* result_ptr){
    device_ptr<int8_t> input_d(input_ptr);
    device_ptr<int8_t> filter_d(filter_ptr);
    device_ptr<int32_t> bias_d(bias_ptr);
    device_ptr<int8_t> result_d(result_ptr);

    constexpr int buff_dim = in_dim+2;
    constexpr int chn_filter_size = in_chn*k_size;
    constexpr int shift_reg_s = 3*buff_dim;

    for (int outc = 0; outc < out_chn; outc++){

        for (int inc = 0; inc < in_chn; inc++){

            [[intel::fpga_register]] int kernel[k_size];
            #pragma unroll
            for (int i = 0; i < k_size; i++){
                kernel[i] = filter_d[outc*chn_filter_size + inc*k_size];
            }

            // initialise shift_reg
            int8_t shift_reg[shift_reg_s];
            #pragma unroll
            for (int i = 0; i < shift_reg_s; i++){
                shift_reg[i] = 0;
            }

            // Load shift_reg with 1st row

            // place new row with padding
            #pragma unroll
            for (int x = 0; x < in_dim; x++){
                shift_reg[2*buff_dim+1+x] = input_d[inc*in_dim*in_dim + x];
            }
            
            // Place rows and convolve previous row
            for (int y = 1; y < in_dim; y++){

                // shift left by one row
                #pragma unroll
                for (int j = 0; j < in_dim; j++){
                    #pragma unroll
                    for (int i = 0; i < shift_reg_s; i++){
                        shift_reg[i] = shift_reg[i+1];
                    }
                }

                // place new row
                shift_reg[2*buff_dim] = 0;
                #pragma unroll
                for (int x = 0; x < in_dim; x++){
                    shift_reg[2*buff_dim+1+x] = input_d[inc*in_dim*in_dim + y*in_dim + x];
                }
                shift_reg[shift_reg_s-1] = 0;

                //calcualte new row
                for (int x = 0; x < in_dim; x++){
                    int32_t sum = bias_d[outc];
                    #pragma unroll
                    for (int y_off = 0; y_off < 3; y_off++){
                        #pragma unroll
                        for (int x_off = 0; x_off < 3; x_off++){
                            sum += shift_reg[y_off*buff_dim + x_off + x] * filter_d[outc * kernel[y_off*3+x_off]];
                        }
                    }
                    result_d[outc*in_dim*in_dim + (y-1)*in_dim + x] = sum > 0 ? round(sum * scale) : 0;
                }
            }

            //Insert final padding layer and convolve last row

            // shift left by one row
            // shift left by one row
                #pragma unroll
                for (int j = 0; j < in_dim; j++){
                    #pragma unroll
                    for (int i = 0; i < shift_reg_s; i++){
                        shift_reg[i] = shift_reg[i+1];
                    }
                }

            // place new row
            #pragma unroll
            for (int x = 0; x < buff_dim; x++){
                shift_reg[2*buff_dim+x] = 0;
            }

            //calcualte new row
            for (int x = 0; x < in_dim; x++){
                int32_t sum = bias_d[outc];
                #pragma unroll
                for (int y_off = 0; y_off < 3; y_off++){
                    #pragma unroll
                    for (int x_off = 0; x_off < 3; x_off++){
                        sum += shift_reg[y_off*buff_dim + x_off + x] * filter_d[outc * kernel[y_off*3+x_off]];
                    }
                }
                result_d[outc*in_dim*in_dim + (in_dim-1)*in_dim + x] = sum > 0 ? sum * scale : 0;
            }

        }
    }
}

template <int chn, int in_dim>
void MaxPool(int8_t *input_ptr, int8_t *result_ptr){
    #pragma clang fp contract(fast)
    device_ptr<int8_t> in_d(input_ptr);
    device_ptr<int8_t> result_d(result_ptr);

    constexpr int out_dim = in_dim / 2;
    for (int c = 0; c < chn; c++){
        for (int y = 0; y < in_dim; y+=2){
            #pragma unroll 2
            for (int x = 0; x < in_dim; x+=2){
                int8_t res = sycl::max(in_d[c*in_dim*in_dim + y*in_dim + x], in_d[c*in_dim*in_dim + y*in_dim + x+ 1]);
                res = sycl::max(res, in_d[c*in_dim*in_dim + (y+1)*in_dim + x]);
                res = sycl::max(res, in_d[c*in_dim*in_dim + (y+1)*in_dim + x + 1]);
                result_d[c*out_dim*out_dim + y/2*out_dim + x/2] = res;
            }
        }
    }
}

#endif