#ifndef __PARAMS_HPP__
#define __PARAMS_HPP__

#include <cinttypes>
#include <fstream>

using namespace std;

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

#endif