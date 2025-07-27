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

__device__ float from_func_Rec709(float v)
{
if (v < 0.08145f) {
return (v < 0.0f) ? 0.0f : v * (1.0f / 4.5f);
} else {
return powf( (v + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f) );
}
}

__device__ float to_func_Rec709(float v)
{
if (v < 0.0181f) {
return (v < 0.0f) ? 0.0f : v * 4.5f;
} else {
return 1.0993f * powf(v, 0.45f) - (1.0993f - 1.f);
}
}

__device__ void rgb709_to_xyz(float r, float g, float b, float *x, float *y, float *z)
{
*x = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
*y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
*z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;
}

__device__ void xyz_to_rgb709(float x, float y, float z, float *r, float *g, float *b)
{
*r =  3.2404542f * x + -1.5371385f * y + -0.4985314f * z;
*g = -0.9692660f * x +  1.8760108f * y +  0.0415560f * z;
*b =  0.0556434f * x + -0.2040259f * y +  1.0572252f * z;
}

__device__ float labf(float x)
{
return ( (x) >= 0.008856f ? ( powf(x, (float)1 / 3) ) : (7.787f * x + 16.0f / 116) );
}

__device__ void xyz_to_lab(float x, float y, float z, float *l, float *a, float *b)
{
const float fx = labf( x / (0.412453f + 0.357580f + 0.180423f) );
const float fy = labf( y / (0.212671f + 0.715160f + 0.072169f) );
const float fz = labf( z / (0.019334f + 0.119193f + 0.950227f) );

*l = 116 * fy - 16;
*a = 500 * (fx - fy);
*b = 200 * (fy - fz);
}

__device__ float labfi(float x)
{
return ( x >= 0.206893f ? (x * x * x) : ( (x - 16.0f / 116) / 7.787f ) );
}

__device__ void lab_to_xyz(float l, float a, float b, float *x, float *y, float *z)
{
const float cy = (l + 16) / 116;

*y = (0.212671f + 0.715160f + 0.072169f) * labfi(cy);
const float cx = a / 500 + cy;
*x = (0.412453f + 0.357580f + 0.180423f) * labfi(cx);
const float cz = cy - b / 200;
*z = (0.019334f + 0.119193f + 0.950227f) * labfi(cz);
}

__device__ void rgb709_to_lab(float r, float g, float b, float *l, float *a, float *b_)
{
float x, y, z;

rgb709_to_xyz(r, g, b, &x, &y, &z);
xyz_to_lab(x, y, z, l, a, b_);
}

__device__ void lab_to_rgb709(float l, float a, float b, float *r, float *g, float *b_)
{
float x, y, z;

lab_to_xyz(l, a, b, &x, &y, &z);
xyz_to_rgb709(x, y, z, r, g, b_);
}

__global__ void d_rec709_to_lab(float* p_Input, float* p_Output, int p_Width, int p_Height)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = ((y * p_Width) + x) * 4;

float r = from_func_Rec709(p_Input[index + 0]);
float g = from_func_Rec709(p_Input[index + 1]);
float bb = from_func_Rec709(p_Input[index + 2]);

float l, a, b;

rgb709_to_lab(r, g, bb, &l, &a, &b);
																								  
p_Output[index + 0] = l / 100.0f;
p_Output[index + 1] = a / 200.0f + 0.5f;
p_Output[index + 2] = b / 200.0f + 0.5f;
}
}

__global__ void d_lab_to_rec709(float* p_Input, int p_Width, int p_Height)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = ((y * p_Width) + x) * 4;

float l = p_Input[index + 0] * 100.0f;
float a = (p_Input[index + 1] - 0.5f) * 200.0f;
float b = (p_Input[index + 2] - 0.5f) * 200.0f;
float r, g, bb;
lab_to_rgb709(l, a, b, &r, &g, &bb);

float R = to_func_Rec709(r);
float G = to_func_Rec709(g);
float BB = to_func_Rec709(bb);
																								  
p_Input[index + 0] = R;
p_Input[index + 1] = G;
p_Input[index + 2] = BB;
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
