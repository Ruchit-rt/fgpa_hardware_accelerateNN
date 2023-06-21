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

/* Carry out MaxPool on the given TENSOR (C * H * W) with a stride of 2. */
void max_pool_q(queue &q, int chn, int row, int col, int8_t *tensor, int8_t *result) {
  const int stride = 2;
  const int nr = 1 + (row - 1) / stride;
  const int nc = 1 + (col - 1) / stride;
  {
    buffer m_buf(tensor, range(chn, row, col));
    buffer r_buf(result, range(chn, nr, nc));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

        h.single_task([=] () {
            for (int c = 0; c < chn; c++){
                for (int y = 0 ; y < nr; y++){
                    for (int x = 0; x < nc; x++){
                        int max_r = (y + 1) * stride;
                        int max_c = (x + 1) * stride;
                        int8_t f = numeric_limits<int8_t>::lowest();
                        for (int i = y * stride; i < max_r; i++) {
                            for (int j = x * stride; j < max_c; j++) {
                                int8_t cur = m[c][i][j];
                                f = f > cur ? f : cur;
                            }
                        }
                        r[c][y][x] = f;
                    }
                }
            }
        
        });
    });
  }
}

// Quantise the input TENSOR with the given SCALE.
void quant(queue &q, int size, float scale, float *tensor, int8_t *result) {
  buffer t_buf(tensor, range(size));
  buffer r_buf(result, range(size));
  q.submit([&](auto &h) {
    accessor t(t_buf, h, read_only);
    accessor r(r_buf, h, write_only);

    h.single_task([=]() {
        for (int i = 0; i < size; i++){
            r[i] = round(t[i] / scale);
        }
    });
  });
}

// Dequantise the input TENSOR with the given SCALE.
void dequant(queue &q, int size, float scale, int8_t *tensor, float *result) {
  buffer t_buf(tensor, range(size));
  buffer r_buf(result, range(size));
  q.submit([&](auto &h) {
    accessor t(t_buf, h, read_only);
    accessor r(r_buf, h, write_only);

    h.single_task([=]() {
        for (int i = 0; i < size; i++){
            r[i] = t[i] * scale;
        }
    });
  });
}

/* Carry out quantised convolution (with padding) between the TENSOR (1 * CHN * W * W) and the FILTER (D * CHN * 3 * 3), then ReLU. */
// void conv_pad_q(queue &q, int chn, int size, int8_t *tensor, int8_t *filter, int d, int32_t *biases, float tensor_scale, float filter_scale, float result_scale, int8_t *result) {
//   {
//     buffer inp_b(inp, range(chn, in_dim, in_dim));
//     buffer f_buf(filter, range(out_chn, in_chn, 3, 3));
//     buffer b_buf(biases, range(out_chn));
//     buffer r_buf(result, range(out_chn, in_dim, in_dim));

//     q.submit([&](auto &h) {
//       accessor i(inp_b, h, read_only);
//       accessor f(f_buf, h, read_only);
//       accessor b(b_buf, h, read_only);
//       accessor r(r_buf, h, write_only);

//       // Task for the corner elements
//       h.single_task([=]() {
//         for (int filter_d = 0; filter_d < d; filter_d++){
//             int32_t sum = 0;
//             const float scale = tensor_scale * filter_scale / result_scale;
//             for (int c = 0; c < chn; c++) {
//             int _fi = filter_d * chn + c;
//             for (int i = 0; i <= 1; i++) {
//                 for (int j = 0; j <= 1; j++) {
//                 sum += f[_fi][i + 1][j + 1] * m[c][i][j];        
//                 }
//             }
//             }

//             sum += b[filter_d];
//             r[filter_d][0][0] = sum > 0 ? round(sum * scale) : 0;

//             sum = 0;
//             for (int c = 0; c < chn; c++) {
//             int _fi = filter_d * chn + c;
//             for (int i = 0; i <= 1; i++) {
//                 for (int j = -2; j <= -1; j++) {
//                 sum += f[_fi][i + 1][j + 2] * m[c][i][size + j];        
//                 }
//             }
//             }

//             sum += b[filter_d];
//             r[filter_d][0][size - 1] = sum > 0 ? round(sum * scale) : 0;

//             sum = 0;
//             for (int c = 0; c < chn; c++) {
//             int _fi = filter_d * chn + c;
//             for (int i = -2; i <= -1; i++) {
//                 for (int j = 0; j <= 1; j++) {
//                 sum += f[_fi][i + 2][j + 1] * m[c][size + i][j];        
//                 }
//             }
//             }

//             sum += b[filter_d];
//             r[filter_d][size - 1][0] = sum > 0 ? round(sum * scale) : 0;

//             sum = 0;
//             for (int c = 0; c < chn; c++) {
//             int _fi = filter_d * chn + c;
//             for (int i = -2; i <= -1; i++) {
//                 for (int j = -2; j <= -1; j++) {
//                 sum += f[_fi][i + 2][j + 2] * m[c][size + i][size + j];        
//                 }
//             }
//             }

//             sum += b[filter_d];
//             r[filter_d][size - 1][size - 1] = sum > 0 ? round(sum * scale) : 0;
//         }
//       });
//     });

//     // Task for the boundary elements.
//     q.submit([&](auto &h) {
//       accessor m(m_buf, h, read_only);
//       accessor f(f_buf, h, read_only);
//       accessor b(b_buf, h, read_only);
//       accessor r(r_buf, h, write_only);

//       h.single_task([=]() {
//         for (int filter_d = 0; filter_d < d; filter_d++){
//             for (int x = 0; x < size - 2; x++){
//                 int32_t sum = 0;
//                 const float scale = tensor_scale * filter_scale / result_scale;
//                 for (int c = 0; c < chn; c++) {
//                 int _fi = filter_d * chn + c;
//                 for (int i = 0; i <= 1; i++) {
//                     for (int j = 0; j <= 2; j++) {
//                     sum += f[_fi][i + 1][j] * m[c][i][x + j];        
//                     }
//                 }
//                 }

//                 sum += b[filter_d];
//                 r[filter_d][0][x + 1] = sum > 0 ? round(sum * scale) : 0;

//                 sum = 0;
//                 for (int c = 0; c < chn; c++) {
//                 int _fi = filter_d * chn + c;
//                 for (int i = -2; i <= -1; i++) {
//                     for (int j = 0; j <= 2; j++) {
//                     sum += f[_fi][i + 2][j] * m[c][size + i][x + j];        
//                     }
//                 }
//                 }

//                 sum += b[filter_d];
//                 r[filter_d][size - 1][x + 1] = sum > 0 ? round(sum * scale) : 0;

//                 sum = 0;
//                 for (int c = 0; c < chn; c++) {
//                 int _fi = filter_d * chn + c;
//                 for (int i = 0; i <= 2; i++) {
//                     for (int j = 0; j <= 1; j++) {
//                     sum += f[_fi][i][j + 1] * m[c][x + i][j];        
//                     }
//                 }
//                 }

//                 sum += b[filter_d];
//                 r[filter_d][x + 1][0] = sum > 0 ? round(sum * scale) : 0;

//                 sum = 0;
//                 for (int c = 0; c < chn; c++) {
//                 int _fi = filter_d * chn + c;
//                 for (int i = 0; i <= 2; i++) {
//                     for (int j = -2; j <= -1; j++) {
//                     sum += f[_fi][i][j + 2] * m[c][x + i][size + j];        
//                     }
//                 }
//                 }

//                 sum += b[filter_d];
//                 r[filter_d][x + 1][size - 1] = sum > 0 ? round(sum * scale) : 0;
//             }
//         }
//       });
//     });

//     // Task for interior elements (that uses all 3 * 3 filters).
//     q.submit([&](auto &h) {
//       accessor m(m_buf, h, read_only);
//       accessor f(f_buf, h, read_only);
//       accessor b(b_buf, h, read_only);
//       accessor r(r_buf, h, write_only);

//       h.single_task([=]() {
//         for (int filter_d = 0; filter_d < d; filter_d++){
//             for (int y = 0; y < size - 2; y++){
//                 for (int x = 0; x < size - 2; x++){
//                     int32_t sum = 0;
//                     const float scale = tensor_scale * filter_scale / result_scale;
//                     for (int c = 0; c < chn; c++) {
//                     int _fi = filter_d * chn + c;
//                     for (int i = 0; i <= 2; i++) {
//                         for (int j = 0; j <= 2; j++) {
//                         sum += f[_fi][i][j] * m[c][y + i][x + j];        
//                         }
//                     }
//                     }

//                     sum += b[filter_d];
//                     r[filter_d][y + 1][x + 1] = sum > 0 ? round(sum * scale) : 0;
//                 }
//             }
//         }
//       });
//     });
//   }
// }

/* Carry out the calculation for a fully-connected layer. */
void fully_connected(queue &q, int c_in, int c_out, int8_t *vector,
    int8_t *weights, int8_t *result) {
  {
    buffer v_buf(vector, range(c_in));
    buffer w_buf(weights, range(c_out, c_in));
    buffer r_buf(result, range(c_out));
    q.submit([&](auto &h) {
      accessor v(v_buf, h, read_only);
      accessor w(w_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.single_task([=]() {
        for (int index = 0; index < c_out; index++){
            // The scales are hardcoded for the sole fully-connected layer of our model.
            const float tensor_scale = 0.01979798823595047;
            const float filter_scale = 0.009601877070963383;
            const float result_scale = 0.06617073714733123779;
            const float scale = tensor_scale * filter_scale / result_scale;
            int32_t sum = 0;
            for (int i = 0; i < c_in; i++) {
            sum += v[i] * w[index][i];
            }
            r[index] = round(sum * scale);
        }
      });
    });
  }
}

/* The L2-distance computation, used for the prototype layer. */
void l2_distance(queue &q, int chn, int length, float *tensor, int d, float *prototypes, float *result) {
  {
    buffer m_buf(tensor, range(chn, length));
    buffer p_buf(prototypes, range(d, chn));
    buffer r_buf(result, range(d, length));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor p(p_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.single_task([=]() {
        for (int p_idx = 0; p_idx < d; p_idx++){
            for (int i = 0; i < length; i++){
                float sum = 0.0f;
                for (int c = 0; c < chn; c++) {
                sum += (m[c][i] - p[p_idx][c]) * (m[c][i] - p[p_idx][c]);
                }
                r[p_idx][i] = sqrt(sum);
            }
        }
      });
    });
  }
}

/* Convert distances to similarity map (part of the prototype layer). */
void distance_2_similarity(queue &q, int length, float *vector, float *result) {
  {
    buffer v_buf(vector, range(length));
    buffer r_buf(result, range(length));
    q.submit([&](auto &h) {
      accessor v(v_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.single_task([=]() {
        for (int index = 0; index < length; index++){
            r[index] = log((v[index] + 1) / (v[index] + 0.0001f));
        }
      });
    });
  }
}

/* Pooling that takes the largest (or smallest, based on IS_TOP) 9 elements, then take the average. */
void top9_average_pooling(queue &q, int chn, int length, float *tensor, float *result, bool is_top) {
  {
    buffer m_buf(tensor, range(chn, length));
    buffer r_buf(result, range(chn));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h);

      // The implementation uses a max-heap to keep track of the 9 largest elements.
      h.single_task([=]() {
        for (int index = 0; index < chn; index++){
            float r_[9];
            for (int i = 0; i < 9; i++) {
            r_[i] = m[index][i];
            int k = i;
            while (true) {
                if (((r_[k] >= r_[k / 2]) && is_top) || ((r_[k] <= r_[k / 2]) && !is_top)) break;
                float temp = r_[k];
                r_[k] = r_[k / 2];
                r_[k / 2] = temp;
                k /= 2;
            }
            }

            for (int i = 9; i < length; i++) {
            if ((m[index][i] > r_[0]) == is_top) {
                r_[0] = m[index][i];
                int k = 0;
                while (k < 9) {
                if (k >= 4) break;

                if ((r_[k] > r_[2 * k + 1]) == is_top || (r_[k] > r_[2 * k + 2]) == is_top) {
                    float temp = r_[k];
                    if ((r_[2 * k + 1] < r_[2 * k + 2]) == is_top) {
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
void upsample4(queue &q, int chn, int row, int col, float *tensor, float *result) {
  {
    buffer m_buf(tensor, range(chn, row, col));
    buffer r_buf(result, range(chn, row * 4, col * 4));

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.single_task([=]() {
        for (int index = 0; index < chn; index++){
            for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++){
                r[index][i][j] = m[index][0][0];
                r[index][i][4 * col - 1 - j] = m[index][0][col - 1];
                r[index][4 * row - 1 - i][j] = m[index][row - 1][0];
                r[index][4 * row - 1 - i][4 * col - 1 - j] = m[index][row - 1][col - 1];
            }
            }
        }
      });
    });

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.single_task([=]() {
        for (int cur_c = 0; cur_c < chn; cur_c++){
            for (int x = 0; x < col * 4 - 4; x++){
                auto _r = 2 * (x % 4) + 1;
                auto _q = x / 4;
                for (int i = 0; i < 2; i++) {
                r[cur_c][i][x + 2] = (m[cur_c][0][_q] * (8 - _r) + m[cur_c][0][_q + 1] * _r) / 8;
                r[cur_c][4 * row - 1 - i][x + 2] = (m[cur_c][row - 1][_q] * (8 - _r) + m[cur_c][row - 1][_q + 1] * _r) / 8;
                }
            }
        }
      });
    });

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.single_task([=]() {
            for (int cur_c = 0; cur_c < chn; cur_c++){
                for (int x = 0; x < row * 4 - 4; x++){
                    auto _r = 2 * (x % 4) + 1;
                    auto _q = x / 4;
                    for (int i = 0; i < 2; i++) {
                    r[cur_c][x + 2][i] = (m[cur_c][_q][0] * (8 - _r) + m[cur_c][_q + 1][0] * _r) / 8;
                    r[cur_c][x + 2][4 * col - 1 - i] = (m[cur_c][_q][col - 1] * (8 - _r) + m[cur_c][_q + 1][col - 1] * _r) / 8;
                    }
                }
            }
      });
    });

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.single_task([=]() {
        for (int cur_c = 0; cur_c < chn; cur_c++){
            for (int y = 0; y < row * 4 - 4; y++){
                for (int x = 0; x < col * 4 - 4; x++){
                    auto _r1 = 2 * (y % 4) + 1;
                    auto _q1 = y / 4;
                    auto _r2 = 2 * (x % 4) + 1;
                    auto _q2 = x / 4;
                    r[cur_c][y + 2][x + 2]
                        = (m[cur_c][_q1][_q2] * (8 - _r1) + m[cur_c][_q1 + 1][_q2] * _r1) * (8 - _r2) / 64
                        + (m[cur_c][_q1][_q2 + 1] * (8 - _r1) + m[cur_c][_q1 + 1][_q2 + 1] * _r1) * _r2 / 64;
                }
            }
        }
      });
    });
  }
}


int main() {
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
    queue q = sycl::queue(selector, queue_properties);

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
  ifstream rf_data("data/model_params_quant.mmzk", ios::binary);
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

  // Timings.
  long times[N] = {};

  for (int i = 0; i < N; i++) {
    auto start = high_resolution_clock::now();

    // Quantise the input.
    quant(q, 3 * 224 * 224, 0.01979798823595047, input_f, input);

    // Convolutional layers (* 2).
    conv_cond(q, 3, 224, 224, input, 64, 3, weights1, biases1, conved1);
    max_pool_q(q, 64, 224, 224, conved1, pooled1);
    conv_cond(q, 64, 112, 112, pooled1, 512, 3, weights2, biases2, conved2);
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

    // Compute min_distance (information for interpretation).
    top9_average_pooling(q, 15, 56 * 56, distances_f, avg_f, false);

    // Compute upsampled activation map (information for interpretation).
    upsample4(q, 15, 56, 56, similarities_f, upsampled_f);

    auto stop = high_resolution_clock::now();
    times[i] = duration_cast<microseconds>(stop - start).count();
  }

  // Print out the output.
//   cout << "\n############################## Input: \n";
//   peek_int(224,224,input, true);
//   cout << "\n############################## Conved1: \n";
//   peek_int(224,224,conved1, true);
//   cout << "\n############################## Pool1: \n";
//   peek_int(112,112,pooled1, true);
//   cout << "\n############################## Conved2: \n";
//   peek_int(112,112,conved2, true);
//   cout << "\n############################## Pool2: \n";
//   peek_int(56,56,pooled2, true);
//   cout << "\n############################## Similarities: \n";
//   peek(56,56,similarities_f, true);
//   cout << "\n############################## Sim Scores: \n";
//   peek_int(1, 15, avg, false);
//   cout << "\n############################## Logits: \n";
//   peek(1, 3, logits_f, true); // The index corresponding to the maximum value is the index of the chosen classification. 0 For cabbage; 1 for carrot; 2 for tomato.
//   cout << "\n############################## Upsample: \n";
//   peek(224, 224, upsampled_f, true);
//   cout << "\n##############################Input: \n";
//   peek(224, 224, upsampled_f + 224 * 224, true);
//   peek(224, 224, upsampled_f + 14 * 224 * 224, true);

  // Output timings.
  long total = 0;
  for (int i = 0; i < N; i++) {
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
  delete[] input;
  delete[] conved1;
  delete[] pooled1;
  delete[] conved2;
  delete[] pooled2;
  delete[] pooled2_f;
  delete[] distances_f;
  delete[] similarities_f;
  delete[] avg_f;
  delete[] logits;
  delete[] logits_f;
  delete[] upsampled_f;

  return 0;
}