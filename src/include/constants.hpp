#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__    
    
constexpr int img_dim = 224;
constexpr int img_pad_dim = 226;
constexpr int c1_dim = 112;
constexpr int c1_pad_dim = 114;
constexpr int c2_dim = 56;

constexpr float c1_scale = 0.01979798823595047*0.013484773226082325/0.04881289601325989;
constexpr float c2_scale = 0.04881289601325989*0.0006907337228767574/0.016132580116391182;

constexpr int img_chn = 3;
constexpr int img_chn_size = img_dim*img_dim;
constexpr int c1_chn_size = c1_dim*c1_dim;
constexpr int img_pad_chn_size = img_pad_dim*img_pad_dim;
constexpr int c1_pad_chn_size = c1_pad_dim*c1_pad_dim;
constexpr int img_size = img_chn*img_dim*img_dim;
constexpr int img_pad_size = img_chn*img_pad_chn_size;

constexpr int filter_size = 9;
constexpr int c1_chn = 64;
constexpr int c2_chn = 512;
constexpr int c2_chn_size = c2_dim*c2_dim;
constexpr int num_protos = 15;
constexpr int num_classes = 3;


constexpr int img_bursts = img_dim/2;
constexpr int conv1_bursts = img_dim;
constexpr int pool1_bursts = c1_dim/2;
constexpr int conv2_bursts = c1_dim;
constexpr int pool2_bursts = c2_dim;
constexpr int dequant_bursts = c2_dim;

constexpr int img_burst_size = 2*img_dim;
constexpr int conv1_burst_size = img_dim;
constexpr int pool1_burst_size = 2*c1_dim;
constexpr int conv2_burst_size = c1_dim;
constexpr int pool2_burst_size = c2_dim;
constexpr int dequant_burst_size = c2_dim;

#endif