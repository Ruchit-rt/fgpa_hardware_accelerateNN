#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <limits>
#include "dpc_common.hpp"
#include <oneapi/dpl/random>
#include <oneapi/mkl.hpp>

#include <sycl/ext/intel/fpga_extensions.hpp>


using namespace std::chrono;
using namespace std;
using namespace sycl;

#define N 50

/* Make a matrix of the given size and initialise all values to x. */
void mk_mat(queue &q, int row, int col, float x, float *result) {
  {
    buffer r_buf(result, range(row, col));
    q.submit([&](auto &h) {
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(row, col), [=](auto index) {
        r[index] = x;
      });
    });
  }
}

/* Make a matrix of the given size and initliase all values randomly between 0 and 1. */
void mk_mat_rand(queue &q, int row, int col, float *result) {
  {
    buffer r_buf(result, range(row, col));
    q.submit([&](auto &h) {
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(row, col), [=](auto index) {
        oneapi::dpl::minstd_rand engine(114, index.get_linear_id());
        oneapi::dpl::uniform_real_distribution<float> distr;
        r[index] = distr(engine);
      });
    });
  }
}

/* Show the matrix. If snapshot = true, only shwo the first 5 * 5 corner. */
void peek(int row, int col, float *matrix, bool snapshot) {
  for (int i = 0; i < (snapshot && row > 5 ? 5 : row); i++) {
    for (int j = 0; j < (snapshot && col > 5 ? 5 : col); j++) {
      cout << std::fixed << matrix[i * col + j] << "\t";
    }
    if (col > 5 && snapshot) {
      cout << "...";
    }
    cout << std::endl;
  }
  if (row > 5 && snapshot) {
    cout << "...";
  }
  cout << std::endl;
}

/* Carry out MaxPool on the given matrices (C * H * W) with the given stride. */
void max_pool(queue &q, int chn, int row, int col, float *matrix, int stride,
    int *new_row, int *new_col, float *result) {
  const int nr = 1 + (row - 1) / stride;
  const int nc = 1 + (col - 1) / stride;
  {
    buffer m_buf(matrix, range(chn, row, col));
    buffer r_buf(result, range(chn, nr, nc));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn, nr, nc), [=](auto index) {
        int _chn = index[0];
        int max_r = (index[1] + 1) * stride > row ? row : (index[1] + 1) * stride;
        int max_c = (index[2] + 1) * stride > col ? col : (index[2] + 1) * stride;
        float f = numeric_limits<float>::lowest();
        float cur;
        for (int i = index[1] * stride; i < max_r; i++) {
          for (int j = index[2] * stride; j < max_c; j++) {
            cur = m[_chn][i][j];
            f = f > cur ? f : cur;
          }
        }
        r[index] = f;
      });
    });
  }

  if (new_row) *new_row = nr;
  if (new_col) *new_col = nc;
}

void relu(queue &q, int length, float *matrix, float *result) {
  {
    buffer m_buf(matrix, range(length));
    buffer r_buf(result, range(length));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(length), [=](auto index) {
        r[index] = r[index] >= 0.0 ? r[index] : 0.0;
      });
    });
  }
}

/* Carry out convolution (no padding) between the tensor (B * C * W * H) and the filters (D * C * K * K).
   size must be an odd number and less than min(row, col). */
void conv(queue &q, int batch, int chn, int row, int col, float *tensor,
    int d, int size, float *filter, float *biases, int *new_row, int *new_col, float *result) {
  const int nr = row - size + 1;
  const int nc = col - size + 1;
  {
    buffer m_buf(tensor, range(batch * chn, row, col));
    buffer f_buf(filter, range(d * chn, size, size));
    buffer b_buf(biases, range(d));
    buffer r_buf(result, range(batch * d, nr, nc));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor f(f_buf, h, read_only);
      accessor b(b_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(batch * d, nr, nc), [=](auto index) {
        float sum = 0.0f;
        int _d = index[0] / batch;
        int _batch = index[0] / d;
        for (int c = 0; c < chn; c++) {
          for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
              sum += f[_d * chn + c][i][j] * m[_batch * chn + c][index[1] + i][index[2] + j];
            }
          }
        }
        r[index] = sum + b[_d];
      });
    });
  }

  if (new_row) *new_row = nr;
  if (new_col) *new_col = nc;
}

/* Carry out convolution (with padding) between the tensor (B * C * W * H) and the filters (D * C * K * K).
   size must be an odd number and less than min(row, col). */
void conv_pad(queue &q, int batch, int chn, int row, int col, float *tensor,
    int d, int size, float *filter, float *biases, float *result) {
  {
    buffer m_buf(tensor, range(batch * chn, row, col));
    buffer f_buf(filter, range(d * chn, size, size));
    buffer b_buf(biases, range(d));
    buffer r_buf(result, range(batch * d, row, col));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor f(f_buf, h, read_only);
      accessor b(b_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(batch * d, row, col), [=](auto index) {
        float sum = 0.0f;
        int _d = index[0] / batch;
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

/* Carry out the calculation for a fully-connected layer. */
void fully_connected(queue &q, int c_in, int c_out, float *vector,
    float *weights, float *bias, float *result) {
  {
    buffer v_buf(vector, range(c_in));
    buffer w_buf(weights, range(c_out, c_in));
    buffer b_buf(bias, range(c_out));
    buffer r_buf(result, range(c_out));
    q.submit([&](auto &h) {
      accessor v(v_buf, h, read_only);
      accessor w(w_buf, h, read_only);
      accessor b(b_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(c_out), [=](auto index) {
        float sum = 0.0f;
        for (int i = 0; i < c_in; i++) {
          sum += v[i] * w[index][i];
        }
        r[index] = sum + b[index];
      });
    });
  }
}

void l2_distance(queue &q, int chn, int length, float *matrices, int d, float *prototypes, float *result) {
  {
    buffer m_buf(matrices, range(chn, length));
    buffer p_buf(prototypes, range(d, chn));
    buffer r_buf(result, range(d, length));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor p(p_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(d, length), [=](auto index) {
        float sum = 0.0f;
        for (int c = 0; c < chn; c++) {
          sum += (m[c][index[1]] - p[index[0]][c]) * (m[c][index[1]] - p[index[0]][c]);
        }
        r[index] = sqrt(sum);
      });
    });
  }
}

void distance_2_similarity(queue &q, int length, float *vector, float *result) {
  {
    buffer v_buf(vector, range(length));
    buffer r_buf(result, range(length));
    q.submit([&](auto &h) {
      accessor v(v_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(length), [=](auto index) {
        r[index] = log((v[index] + 1) / (v[index] + 0.0001));
      });
    });
  }
}

void top9(queue &q, int chn, int length, float *tensor, float *result) {
  {
    buffer m_buf(tensor, range(chn, length));
    buffer r_buf(result, range(chn, 9));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h);

      h.parallel_for(range(chn), [=](auto index) {
        for (int i = 0; i < 9; i++) {
          r[index][i] = m[index][i];
          int k = i;
          while (true) {
            if (r[index][k] >= r[index][k / 2]) break;
            float temp = r[index][k];
            r[index][k] = r[index][k / 2];
            r[index][k / 2] = temp;
            k /= 2;
          }
        }

        for (int i = 9; i < length; i++) {
          if (m[index][i] > r[index][0]) {
            r[index][0] = m[index][i];
            int k = 0;
            while (k < 9) {
              if (k >= 4) break;

              if (r[index][k] > r[index][2 * k + 1] || r[index][k] > r[index][2 * k + 2]) {
                float temp = r[index][k];
                if (r[index][2 * k + 1] < r[index][2 * k + 2]) {
                  r[index][k] = r[index][2 * k + 1];
                  r[index][2 * k + 1] = temp;
                  k = 2 * k + 1;
                } else {
                  r[index][k] = r[index][2 * k + 2];
                  r[index][2 * k + 2] = temp;
                  k = 2 * k + 2;
                }
              } else {
                break;
              }
            }
          }
        }
      });
    });
  }
}

void bottom9(queue &q, int chn, int length, float *tensor, float *result) {
  {
    buffer m_buf(tensor, range(chn, length));
    buffer r_buf(result, range(chn, 9));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h);

      h.parallel_for(range(chn), [=](auto index) {
        for (int i = 0; i < 9; i++) {
          r[index][i] = m[index][i];
          int k = i;
          while (true) {
            if (r[index][k] <= r[index][k / 2]) break;
            float temp = r[index][k];
            r[index][k] = r[index][k / 2];
            r[index][k / 2] = temp;
            k /= 2;
          }
        }

        for (int i = 9; i < length; i++) {
          if (m[index][i] < r[index][0]) {
            r[index][0] = m[index][i];
            int k = 0;
            while (k < 9) {
              if (k >= 4) break;

              if (r[index][k] < r[index][2 * k + 1] || r[index][k] < r[index][2 * k + 2]) {
                float temp = r[index][k];
                if (r[index][2 * k + 1] > r[index][2 * k + 2]) {
                  r[index][k] = r[index][2 * k + 1];
                  r[index][2 * k + 1] = temp;
                  k = 2 * k + 1;
                } else {
                  r[index][k] = r[index][2 * k + 2];
                  r[index][2 * k + 2] = temp;
                  k = 2 * k + 2;
                }
              } else {
                break;
              }
            }
          }
        }
      });
    });
  }
}

void average(queue &q, int chn, int length, float *tensor, float *result) {
  {
    buffer m_buf(tensor, range(chn, length));
    buffer r_buf(result, range(chn));

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn), [=](auto index) {
        float sum = 0;
        for (int i = 0; i < length; i++) {
          sum += m[index][i];
        }
        r[index] = sum / length;
      });
    });
  }
}

void upsample4(queue &q, int chn, int row, int col, float *matrix, float *result) {
  {
    buffer m_buf(matrix, range(chn, row, col));
    buffer r_buf(result, range(chn, row * 4, col * 4));

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn), [=](auto index) {
        auto _r = 2 * (index[1] % 8) + 1;
        auto _q = (index[1] + 2) / 4;
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++){
            r[index][i][j] = m[index][0][0];
            r[index][i][4 * col - 1 - j] = m[index][0][col - 1];
            r[index][4 * row - 1 - i][j] = m[index][row - 1][0];
            r[index][4 * row - 1 - i][4 * col - 1 - j] = m[index][row - 1][col - 1];
          }
        }
      });
    });

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn, col * 4 - 4), [=](auto index) {
        auto _r = 2 * (index[1] % 8) + 1;
        auto _q = (index[1] + 2) / 4;
        for (int i = 0; i < 2; i++) {
          r[index[0]][i][index[1] + 2] = (m[index[0]][0][_q] * (8 - _r) + m[index[0]][0][_q + 1] * _r) / 8;
          r[index[0]][4 * row - 1 - i][index[1] + 2] = (m[index[0]][row - 1][_q] * (8 - _r) + m[index[0]][row - 1][_q + 1] * _r) / 8;
          }
      });
    });

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn, row * 4 - 4), [=](auto index) {
        auto _r = 2 * (index[1] % 8) + 1;
        auto _q = (index[1] + 2) / 4;
        for (int i = 0; i < 2; i++) {
          r[index[0]][index[1] + 2][i] = (m[index[0]][_q][0] * (8 - _r) + m[index[0]][_q + 1][0] * _r) / 8;
          r[index[0]][index[1] + 2][4 * col - 1 - i] = (m[index[0]][_q][col - 1] * (8 - _r) + m[index[0]][_q + 1][col - 1] * _r) / 8;
        }
      });
    });

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn, row * 4 - 4, col * 4 - 4), [=](auto index) {
        auto _r1 = 2 * (index[1] % 4) + 1;
        auto _q1 = index[1] / 4;
        auto _r2 = 2 * (index[2] % 4) + 1;
        auto _q2 = index[2] / 4;
        r[index[0]][index[1] + 2][index[2] + 2]
            = (m[index[0]][_q1][_q2] * (8 - _r1) + m[index[0]][_q1 + 1][_q2] * _r1) * (8 - _r2) / 64
            + (m[index[0]][_q1][_q2 + 1] * (8 - _r1) + m[index[0]][_q1 + 1][_q2 + 1] * _r1) * _r2 / 64;
      });
    });
  }
}

float *read_param(ifstream &rf) {
  int len;
  rf.read((char *)(&len), 4);
  float *result = new float[len];
  rf.read((char *)result, len * 4);
  cout << "Read " << len << " params" << std::endl;
  return result;
}

int main() {
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
    queue q = sycl::queue(selector, queue_properties);

    auto device = q.get_device();

  ifstream rf("data/model_params.mmzk", ios::binary);
  if(!rf) {
    cout << "Cannot open file!" << std::endl;
    return 1;
  }

  /* reading input from file ----- */ 
  std::ifstream inputFile("src/input.txt");
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


  float *weights1 = read_param(rf);
  float *biases1 = read_param(rf);
  float *weights2 = read_param(rf);
  float *biases2 = read_param(rf);
  float *prototypes = read_param(rf);
  float *fc_weights = read_param(rf);
  float fc_biases[3] = {0.0, 0.0, 0.0};

  rf.close();

  float *conved1 = new float[64 * 224 * 224];
  float *pooled1 = new float[64 * 112 * 112];
  float *conved2 = new float[512 * 112 * 112];
  float *pooled2 = new float[512 * 56 * 56];
  float *distances = new float[15 * 56 * 56];
  float *similarities = new float[15 * 56 * 56];
  float *top_similarities = new float[15 * 9];
  float *avg = new float[15];
  float *logits = new float[3];

  long times[N] = {};

  for (int i = 0; i < N; i++) {
    auto start = high_resolution_clock::now();

    conv_pad(q, 1, 3, 224, 224, input_ff, 64, 3, weights1, biases1, conved1);
    relu(q, 64 * 224 * 224, conved1, conved1);
    max_pool(q, 64, 224, 224, conved1, 2, NULL, NULL, pooled1);
    conv_pad(q, 1, 64, 112, 112, pooled1, 512, 3, weights2, biases2, conved2);
    relu(q, 512 * 112 * 112, conved2, conved2);
    max_pool(q, 512, 112, 112, conved2, 2, NULL, NULL, pooled2);
    l2_distance(q, 512, 56 * 56, pooled2, 15, prototypes, distances);
    distance_2_similarity(q, 15 * 56 * 56, distances, similarities);
    top9(q, 15, 56 * 56, similarities, top_similarities);
    average(q, 15, 9, top_similarities, avg);
    fully_connected(q, 15, 3, avg, fc_weights, fc_biases, logits);
    bottom9(q, 15, 56 * 56, distances, top_similarities);
    average(q, 15, 9, top_similarities, avg);
    upsample4(q, 15, 56, 56, similarities, conved1);

    auto stop = high_resolution_clock::now();
    times[i] = duration_cast<microseconds>(stop - start).count();
  }

  peek(1, 3, logits, false);
  peek(1, 15, avg, false);
  peek(224, 224, conved1, true);
  peek(224, 224, conved1 + 224 * 224, true);
  peek(224, 224, conved1 + 14 * 224 * 224, true);

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
  delete[] conved1;
  delete[] pooled1;
  delete[] conved2;
  delete[] pooled2;
  delete[] prototypes;
  delete[] distances;
  delete[] similarities;
  delete[] top_similarities;
  delete[] fc_weights;
  delete[] logits;
  delete[] input_ff;

  return 0;

  }
  else {
   std::cout << "Failed to open the input file." << std::endl;
   return 1;
  }
}