#include "fullyConnected.h"

FullyConnectedLayer::FullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int inSize, int outSize, bool activate) 
    : cl(ConvLayer(hndl, bs, inSize, outSize, 1, 1, 1, 1, activate)),
      in(inSize),
      out(outSize) {}

int FullyConnectedLayer::loadWeights(float* weights) {
    float* wT = (float*) malloc((in * out + out) * sizeof(float));

    for (int i = 0; i < in; i++)
        for (int j = 0; j < out; j++) {
            int i1 = i * out + j;
            int i2 = j * in + i;

            wT[i2] = weights[i1];
        }

    for (int i = in * out; i < in * out + out; i++)
        wT[i] = weights[i];

    int ret = cl.loadWeights(wT);

    free(wT);

    return ret;
}

float* FullyConnectedLayer::forward(float* d_input) {
    return cl.forward(d_input);
}