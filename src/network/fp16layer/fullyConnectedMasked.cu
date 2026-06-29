#include "fullyConnectedMasked.h"
#include "../util.h"

// ai slop for the moment
__global__ void fcfwbatchedsparseout(int batchSize, int in, int* out, int nOut, int outSize,
                                      __half* input, float* output, __half* weights, __half* biases) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int batch    = threadId / nOut;
    if (batch >= batchSize)
        return;
    int idx    = threadId % nOut;
    int outIdx = out[idx + nOut * batch];   // unchanged: out's indexing is independent of input layout
    if (outIdx == -1)
        return;

    int inBase = batch * in;   // in == C*H*W, batch stride is the same total size either way

    float acc = __half2float(biases[outIdx]);

    // Walk h, w, c in that order so the innermost loop (c) reads input
    // contiguously — that's the whole point of switching to NHWC.
    for (int h = 0; h < 8; h++) {
        int wRowOff   = h * 8;            // h*W term, reused for both i and the NHWC offset
        for (int w = 0; w < 8; w++) {
            int spatial = wRowOff + w;     // h*W + w
            int inOff   = inBase + spatial * 8;  // start of this pixel's channel run in NHWC
            int iBase   = spatial;          // i = c*H*W + h*W + w -> c*H*W + iBase
            for (int c = 0; c < 8; c++) {
                int i = c * 8 * 8 + iBase;  // logical NCHW-order index, matches weights' row order
                acc +=
                    __half2float(__hmul(weights[i * outSize + outIdx], input[inOff + c]));
            }
        }
    }

    output[batch * nOut + idx] = acc;
}

MaskedFullyConnectedLayer::MaskedFullyConnectedLayer(const cudnnHandle_t& hndl, int bs, int is, int os) : handle(hndl), batchSize(bs), inSize(is), outSize(os) {
    cudaMalloc(&d_weights, inSize * outSize * sizeof(__half));
    cudaMalloc(&d_biases, outSize * sizeof(__half));
    cudaMalloc(&d_output, 218 * batchSize * sizeof(float));
}

int MaskedFullyConnectedLayer::loadWeights(float* weights) {
    int nWeights = inSize * outSize;

    copyConvertToDevice(d_weights, weights, nWeights);
    copyConvertToDevice(d_biases, weights + nWeights, outSize);

    //convertToOHWC(d_weights, outSize, inSize / 64, 8, 8);

    return nWeights + outSize;
}

float* MaskedFullyConnectedLayer::forward(__half* d_input, int* d_mask) {
    int threads = 256;
    int blocks = ceildiv(batchSize * 218, threads);

    cudaStream_t stream; 
    
    cudnnGetStream(handle, &stream);

    fcfwbatchedsparseout<<<blocks, threads, 0, stream>>>(batchSize, 
                                                         inSize, 
                                                         d_mask, 
                                                         218, 
                                                         outSize, 
                                                         d_input, 
                                                         d_output, 
                                                         d_weights, 
                                                         d_biases);

    return d_output;
}