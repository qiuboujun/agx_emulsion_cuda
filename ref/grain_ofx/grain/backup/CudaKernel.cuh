#ifndef _CUDAKERNEL_CUH_
#define _CUDAKERNEL_CUH_

__global__ void ConvertToFloat4Kernel(int p_Width, int p_Height, float* p_Input, float4* p_Output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < p_Width) && (y < p_Height))
    {
        int index = y * p_Width + x;
        float4 pixel;
        pixel.x = p_Input[index * 4];     // R
        pixel.y = p_Input[index * 4 + 1]; // G
        pixel.z = p_Input[index * 4 + 2]; // B
        pixel.w = p_Input[index * 4 + 3]; // A
        p_Output[index] = pixel;
    }
}

__global__ void GlobalBlend(int p_Width, int p_Height, float* p_Input, float* p_Output, float blend)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
p_Output[index + 0] = p_Output[index + 0] * blend + p_Input[index + 0] * (1.0f - blend);
p_Output[index + 1] = p_Output[index + 1] * blend + p_Input[index + 1] * (1.0f - blend);
p_Output[index + 2] = p_Output[index + 2] * blend + p_Input[index + 2] * (1.0f - blend);
p_Output[index + 3] = p_Output[index + 3] * blend + p_Input[index + 3] * (1.0f - blend);
}
}

#endif
