#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__    
    
constexpr int img_dim = 224;
constexpr int c1_dim = img_dim/2;
constexpr int c2_dim = img_dim/4;

constexpr float c1_scale = 0.01979798823595047*0.013484773226082325/0.04881289601325989;
constexpr float c2_scale = 0.04881289601325989*0.0006907337228767574/0.016132580116391182;

constexpr int img_chn = 3;
constexpr int img_size = img_chn*img_dim*img_dim;    

constexpr int filter_size = 9;
constexpr int c1_chn = 64;
constexpr int c2_chn = 512;
constexpr int num_protos = 15;
constexpr int num_classes = 3;

#endif