#include "CudaKernel.h"
#include "CudaKernel.cuh"
#include <math.h>

/* Texture declaration */
texture<float4,2,cudaReadModeElementType> tex_src_im;
texture <float,1,cudaReadModeElementType> tex_lambda_list;
texture <float,1,cudaReadModeElementType> tex_exp_lambda_list;

/*********************************************/
/*****   PSEUDO-RANDOM NUMBER GENERATOR   ****/
/*********************************************/

/* 
 * From http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
 * Same strategy as in Gabor noise by example
 * Apply hashtable to create cellseed
 * Use a linear congruential generator as fast PRNG
 */
/**
 * 
 */
 /**
* @brief Produce random seed
*
* @param input seed
* @return output, modified seed
*/
__device__
static unsigned int wang_hash(unsigned int input) 
{
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__
static unsigned int cellseed(unsigned int x, unsigned int y, unsigned int offset) 
{
    const unsigned int period = 65536u; // 65536 = 2^16
    unsigned int s = ((y % period) * period + (x % period)) + offset;
    if (s == 0u) s = 1u;
    return s;
}

__device__ unsigned int myrand(unsigned int  *p)
{
// linear congruential generator: procudes correlated output. Similar patterns are visible
// p.state = 1664525u * p.state + 1013904223u;
// Xorshift algorithm from George Marsaglia's paper
    *p ^= (*p << 13u);
    *p ^= (*p >> 17u);
    *p ^= (*p << 5u);
    return(*p);
}

__device__ void mysrand(unsigned int  *p, const unsigned int seed)
{
    unsigned int s=seed;
    *p = wang_hash(s);
}

__device__
static float myrand_uniform_0_1(unsigned int  *p)
{
    return (float) myrand(p) / (float) 4294967295u;
}

__device__ float myrand_gaussian_0_1(unsigned int  *p)
{
    /* Box-Muller method for generating standard Gaussian variate */
    float u = myrand_uniform_0_1(p);
    float v = myrand_uniform_0_1(p);
    return( sqrt(-2.0 * log(u)) * cos(2.0 * pi * v) );
}

__device__
static int my_rand_poisson(unsigned int *prngstate, float lambda, float prod)
{
    /* Inverse transform sampling */
    float u = myrand_uniform_0_1(prngstate);

    float sum = prod;
    float x = 0.0f;
    while ((u > sum) && (x < floorf(10000.0f * lambda))) {
        x += 1.0f;
        prod *= lambda / x;
        sum += prod;
    }

    return (int) x;
}



/*********************************************/
/*********     GRAIN RENDERING     ***********/
/*********************************************/

/**
 * 
 */
 /**
* @brief Square distance 
*
* @param lambda parameter of the Poisson process
* @param x1, y1 : x, y coordinates of the first point
* @param x2, y2 : x, y coordinates of the second point
* @return squared Euclidean distance
*/
__device__
static float sqDistance(float x1, float y1, float x2, float y2) 
{
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}


__global__ void grainKernel(int p_Width, int p_Height, float p_muR, float p_sigmaR, float p_NmonteCarlo, float p_sigmaFilter, int p_seed, float* p_Input, float* p_Output)
{
    __syncthreads();
    // map from blockIdx to pixel position
    float x = (float)(threadIdx.x + blockIdx.x * blockDim.x);
    float y = (float)(threadIdx.y + blockIdx.y * blockDim.y);

	float normalQuantile = 3.0902;//2.3263;	//standard normal quantile for alpha=0.999
	float logNormalQuantile;
	float grainRadiusSq = p_muR*p_muR;
	float currRadius,currGrainRadiusSq;
	float mu, sigma, sigmaSq;
	float maxRadius = p_muR;
	float ag = 1/ceil(1/p_muR);
    float sX = ((float)(p_Width-1))/((float)(p_Width)); 
    float sY = ((float)(p_Height-1))/((float)(p_Height));
    
    //calculate the mu and sigma for the lognormal distribution
	if (p_sigmaR > 0.0)
	{
		sigma = sqrt(log((p_sigmaR/p_muR)*(p_sigmaR/p_muR) + (float)1.0));
		sigmaSq = sigma*sigma;
		mu = log(p_muR)-sigmaSq/((float)2.0);
		logNormalQuantile = exp(mu + sigma*normalQuantile);
		maxRadius = logNormalQuantile;
	}
	 if( (x<p_Width) && (y<p_Height) )
    {
		float pixOut=0.0, u;
		//unsigned int offsetRand = 2;

		//conversion from output grid (xOut,yOut) to input grid (xIn,yIn)
		//we inspect the middle of the output pixel (1/2)
		//the size of a pixel is (xB-xA)/nOut
		x = (x+(float)0.5) * ((float)(p_Width)/((float)p_Width));
		y = (y+(float)0.5) * ((float)(p_Height)/((float)p_Height));

		// Simulate Poisson process on the 4 neighborhood cells of (x,y)
		unsigned int pMonteCarlo;
		unsigned int p;
		mysrand(&pMonteCarlo, ((unsigned int)2023)*(p_seed));

		for (int i=0; i<p_NmonteCarlo; i++)
		{

			float xGaussian = myrand_gaussian_0_1(&pMonteCarlo);
			float yGaussian = myrand_gaussian_0_1(&pMonteCarlo);

			xGaussian = x + p_sigmaFilter*(xGaussian)/sX;
			yGaussian = y + p_sigmaFilter*(yGaussian)/sY;

			// Compute the Poisson parameters for the pixel that contains (x,y)
			/*float4 src_im_sxsy = tex2D(tex_src_im, (int)max(floor(xGaussian),0.0), (int)max(floor(yGaussian),0.0));
			u = src_im_sxsy.x;
			u = u/(uMax+epsilon);
			lambda = -((ag*ag)/( pi*(grainRadiusSq + grainSigma*grainSigma) )) * log(1.0f-u);*/

			//determine the bounding boxes around the current shifted pixel
			// these operations are set to float precision because the number of cells can be quite large
			unsigned int minX = (unsigned int)floor( ( (float)xGaussian - (float)maxRadius)/((float)ag));
			unsigned int maxX = (unsigned int)floor( ( (float)xGaussian + (float)maxRadius)/((float)ag));
			unsigned int minY = (unsigned int)floor( ( (float)yGaussian - (float)maxRadius)/((float)ag));
			unsigned int maxY = (unsigned int)floor( ( (float)yGaussian + (float)maxRadius)/((float)ag));

			bool ptCovered = false; // used to break all for loops

			for(unsigned int ncx = minX; ncx <= maxX; ncx++) // x-cell number
			{
				if(ptCovered == true)
				break;
				for(unsigned int ncy = minY; ncy <= maxY; ncy++) // y-cell number
				{
					if(ptCovered == true)
						break;
					double cellCornerX = ((float)ag)*((float)ncx);
			        double cellCornerY = ((float)ag)*((float)ncy);

					unsigned int seed = cellseed(ncx, ncy, p_seed);
					mysrand(&p,seed);

					// Compute the Poisson parameters for the pixel that contains (x,y)
					float4 src_im_sxsy = tex2D(tex_src_im, (int)max(floor(cellCornerX),0.0), (int)max(floor(cellCornerY),0.0));
					u = src_im_sxsy.x;
                    int uInd = (int)floorf(u * ((float)MAX_FLOAT_LEVELS + (float)EPSILON_GREY_LEVEL));
					float currLambda = tex1D(tex_lambda_list,uInd);
					float currExpLambda = tex1D(tex_exp_lambda_list,uInd);

					/*float currLambda = lambda;
					float currExpLambda = exp(-lambda);
					if((floor(cellCornerX) != floor(xGaussian)) || (floor(cellCornerY) != floor(yGaussian)))
					{
						float4 src_im_temp =
						tex2D(tex_src_im, (int)max(floor(cellCornerX),0.0), (int)max(floor(cellCornerY),0.0));
						// Compute the Poisson parameters for the pixel that contains (x,y)
						u = src_im_temp.x;
						u = u/(uMax+epsilon);
						currLambda = -((ag*ag)/( pi*(grainRadiusSq + grainSigma*grainSigma))) * log(1.0f-u);
						currLambda = exp(-lambda);
					}*/

					unsigned int Ncell = my_rand_poisson(&p, currLambda,currExpLambda);

					for(unsigned int k=0; k<Ncell; k++)
					{
						//draw the grain centre
						//changed to float precision to avoid incorrect operations
						double xCentreGrain = (float)cellCornerX + ((float)ag)*((float)myrand_uniform_0_1(&p));
				        double yCentreGrain = (float)cellCornerY + ((float)ag)*((float)myrand_uniform_0_1(&p));

						//draw the grain radius
						if (p_sigmaR>0.0)
						{
							//draw a random Gaussian radius, and convert it to log-normal
							currRadius = (float)fmin((float)exp(mu + sigma*myrand_gaussian_0_1(&p)),maxRadius);
							currGrainRadiusSq = currRadius*currRadius;
						}
						else
							currGrainRadiusSq = grainRadiusSq;

						// test distance
						if(sqDistance(xCentreGrain, yCentreGrain, xGaussian, yGaussian) < (float)currGrainRadiusSq)
						{
							pixOut = pixOut+(float)1.0;
							ptCovered = true;
							break;
						}
					}
				} 	//end ncy
			}		//end ncx
			ptCovered = false;
		}		//end monte carlo

		// store output
		pixOut = pixOut/((float)p_NmonteCarlo);//lambda;//

		// map from blockIdx to pixel position
		x =  (threadIdx.x + blockIdx.x * blockDim.x);
		y =  (threadIdx.y + blockIdx.y * blockDim.y);

		int index = (int)(x + y * p_Width) * 4;

		p_Output[index] = pixOut;
    }
}

__host__ void getTextureMap(float *p_Input, float* p_Output, int p_Width, int p_Height, float p_muR, float p_sigmaR, float p_NmonteCarlo, float p_sigmaFilter, int p_seed)
{
    /* copy src image on device */
    /* add unnecessary alpha channel to fit the float4 format of texture memory */

    float4* src_imf4;
    int numPixels = p_Width * p_Height;
    cudaMalloc(&src_imf4, numPixels * sizeof(float4)); 
    dim3 threads(16, 16);
    dim3 blocks((p_Width + threads.x - 1) / threads.x, (p_Height + threads.y -1) / threads.y);
    ConvertToFloat4Kernel<<<blocks, threads>>>(p_Width, p_Height, p_Input, src_imf4);

    /* copy input float4 texture on device texture memory */
    cudaArray* dev_src_im;
    cudaChannelFormatDesc descchannel;      
    descchannel=cudaCreateChannelDesc<float4>();
    cudaMallocArray(&dev_src_im, &descchannel, p_Width, p_Height);
    cudaMemcpyToArray(dev_src_im,0,0,src_imf4,
                                 sizeof(float4)*p_Width*p_Height,
                                 cudaMemcpyHostToDevice);
    tex_src_im.filterMode=cudaFilterModePoint;
    tex_src_im.addressMode[0]=cudaAddressModeClamp;
    tex_src_im.addressMode[1]=cudaAddressModeClamp;
    cudaBindTextureToArray(tex_src_im,dev_src_im);

	/*pre-calculate the Gaussian , lambda, and exp(-lambda) */
	//pre-calculate lambda and exp(-lambda) for each possible grey-level
	
	float *lambdaList = new float[MAX_FLOAT_LEVELS + 1];
	float *expLambdaList = new float[MAX_FLOAT_LEVELS + 1];
	for (int i=0; i<=MAX_FLOAT_LEVELS; i++)
	{
		float u = ((float)i)/( (float) ( (float)MAX_FLOAT_LEVELS + (float)EPSILON_GREY_LEVEL) );
		float ag = 1/ceil(1/(p_muR));
		float lambdaTemp = -((ag*ag) /
			( pi*( (p_muR) * (p_muR) +
				(p_sigmaR) * (p_sigmaR)))) * log(1.0f-u);
		lambdaList[i] = lambdaTemp;
		expLambdaList[i] = exp(-lambdaTemp);
	}
    cudaArray* dev_lambda_list, *dev_exp_lambda_list;
    cudaChannelFormatDesc descchannel1D;      
    descchannel1D=cudaCreateChannelDesc<float>();
    cudaMallocArray(&dev_lambda_list, &descchannel1D, MAX_FLOAT_LEVELS+1);
    cudaMallocArray(&dev_exp_lambda_list, &descchannel1D, MAX_FLOAT_LEVELS+1);
    cudaMemcpyToArray(dev_lambda_list,0,0,lambdaList, sizeof(float)*(MAX_FLOAT_LEVELS+1), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(dev_exp_lambda_list,0,0,expLambdaList, sizeof(float)*(MAX_FLOAT_LEVELS+1), cudaMemcpyHostToDevice);
    
    tex_lambda_list.filterMode=cudaFilterModePoint;
    tex_lambda_list.addressMode[0]=cudaAddressModeClamp;
    tex_lambda_list.addressMode[1]=cudaAddressModeClamp;
    cudaBindTextureToArray(tex_lambda_list,dev_lambda_list);
    //exp(-lambda)
    tex_exp_lambda_list.filterMode=cudaFilterModePoint;
    tex_exp_lambda_list.addressMode[0]=cudaAddressModeClamp;
    tex_exp_lambda_list.addressMode[1]=cudaAddressModeClamp ;
    cudaBindTextureToArray(tex_exp_lambda_list,dev_exp_lambda_list);

    grainKernel<<<blocks, threads>>>(p_Width, p_Height, p_muR, p_sigmaR, p_NmonteCarlo, p_sigmaFilter, p_seed, p_Input, p_Output);
    
     /* free memory */
    cudaUnbindTexture(tex_src_im);
    cudaUnbindTexture(tex_lambda_list);
    cudaUnbindTexture(tex_exp_lambda_list);
    cudaFreeArray(dev_lambda_list);
    cudaFreeArray(dev_exp_lambda_list);
    cudaFreeArray(dev_src_im);
    cudaFree(src_imf4);
}

extern void RunCudaKernel(int p_Width, int p_Height, float p_muR, float p_sigmaR, float p_NmonteCarlo, float p_sigmaFilter, int p_seed, float p_blend, float* p_Input, float* p_Output, int p_rnd)
{
    dim3 threads(16, 16);
    dim3 blocks((p_Width + threads.x - 1) / threads.x, (p_Height + threads.y -1) / threads.y);
    //cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);
    d_rec709_to_lab<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height);
    getTextureMap(p_Output + 0, p_Output + 0, p_Width, p_Height, p_muR, p_sigmaR, p_NmonteCarlo, p_sigmaFilter, p_seed*2023*p_rnd);
    d_lab_to_rec709<<<blocks, threads>>>(p_Output, p_Width, p_Height);
    GlobalBlend<<<blocks, threads>>>(p_Width, p_Height, p_Input, p_Output, p_blend);
}
