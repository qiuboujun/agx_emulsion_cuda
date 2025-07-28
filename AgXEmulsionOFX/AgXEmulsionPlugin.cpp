#include "AgXEmulsionPlugin.h"

#include <stdio.h>
#include <random>
#include <iostream>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstring>
#include "EmulsionKernel.cuh"

#define kPluginName "AgX Emulsion"
#define kPluginGrouping "OpenFX JQ"
#define kPluginDescription "Apply AgX film emulation to RGB channels"
#define kPluginIdentifier "com.JQ.AgXEmulsion"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

class AgXEmulsionProcessor : public OFX::ImageProcessor
{
public:
    explicit AgXEmulsionProcessor(OFX::ImageEffect& p_Instance);

    virtual void processImagesCUDA();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    // void setParams(float p_invertStrength);
    void setFrameTime(double frameTime) {_frameTime = frameTime; }
    void setFilmGamma(const std::string& film,float gamma){_film=film;_gamma=gamma;}
    const std::string& getFilm() const {return _film;}
    float getGamma() const {return _gamma;}
private:
    OFX::Image* _srcImg;
    // float _unused;
    double _frameTime;
    std::string _film;
    float _gamma;
};

AgXEmulsionProcessor::AgXEmulsionProcessor(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

extern "C" void LaunchEmulsionCUDA(float* img, int width, int height, float gamma);

void AgXEmulsionProcessor::processImagesCUDA()
{
    static char sFilm[64]="";
    static float sGamma=-1.0f;
    bool lutOK=true;
    if(_film!="" && (strcmp(_film.c_str(),sFilm)!=0 || _gamma!=sGamma)){
        static void* helper = nullptr;
        static bool (*loadFn)(const char*,float*,float*,float*,float*) = nullptr;
        if(!helper){
            helper = dlopen("libAgXLUT.so", RTLD_LAZY);
            if(helper) loadFn = (bool(*)(const char*,float*,float*,float*,float*))dlsym(helper,"loadFilmLUT");
        }
        if(loadFn){
            float logE[601], r[601], g[601], b[601];
            if(loadFn && loadFn(_film.c_str(),logE,r,g,b)){
                UploadLUTCUDA(logE,r,g,b);
                strncpy(sFilm,_film.c_str(),63); sFilm[63]=0;
                sGamma=_gamma;
            } else {
                lutOK=false;
            }
        }
    }
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width  = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* srcPtr = static_cast<float*>(_srcImg->getPixelData());
    float* dstPtr = static_cast<float*>(_dstImg->getPixelData());
    size_t bytes   = static_cast<size_t>(width) * height * 4 * sizeof(float);

    // Copy input image to output on the device so we can modify in-place
    cudaMemcpy(dstPtr, srcPtr, bytes, cudaMemcpyDeviceToDevice);

    if(lutOK){
        // Launch emulsion kernel
        LaunchEmulsionCUDA(dstPtr, width, height, _gamma);
    }
}

void AgXEmulsionProcessor::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            // do we have a source image to scale up
            if (srcPix)
            {
                dstPix[0] = srcPix[0];
                dstPix[1] = srcPix[1];
                dstPix[2] = srcPix[2];
                dstPix[3] = srcPix[3];
            }
            else
            {
                // no src pixel here, be black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }

            // increment the dst pixel
            dstPix += 4;
        }
    }
}

void AgXEmulsionProcessor::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}


////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class AgXEmulsionPlugin : public OFX::ImageEffect
{
public:
    explicit AgXEmulsionPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Set up and run a processor */
    void setupAndProcess(AgXEmulsionProcessor &p_AgXProcessor, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::ChoiceParam* m_filmStock;
    OFX::DoubleParam* m_gammaFactor;
};

AgXEmulsionPlugin::AgXEmulsionPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    m_filmStock = fetchChoiceParam("filmStock");
    m_gammaFactor = fetchDoubleParam("gammaFactor");
}

void AgXEmulsionPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        AgXEmulsionProcessor agxProcessor(*this);
        agxProcessor.setFrameTime(p_Args.time);
        setupAndProcess(agxProcessor, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool AgXEmulsionPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    if (m_SrcClip)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }
    return false;
}

void AgXEmulsionPlugin::setupAndProcess(AgXEmulsionProcessor& p_AgXProcessor, const OFX::RenderArguments& p_Args)
{
    // Get the dst image
    std::auto_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the src image
    std::auto_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    // Check to see if the bit depth and number of components are the same
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }

    int filmIdx; m_filmStock->getValueAtTime(p_Args.time, filmIdx);
    std::string filmName = filmIdx==0?"kodak_portra_400":"kodak_vision3_250d";
    double gamma = m_gammaFactor->getValueAtTime(p_Args.time);
    
    // Set the images
    p_AgXProcessor.setDstImg(dst.get());
    p_AgXProcessor.setSrcImg(src.get());
  
    // Setup OpenCL and CUDA Render arguments
    p_AgXProcessor.setGPURenderArgs(p_Args);

    // Set the render window
    p_AgXProcessor.setRenderWindow(p_Args.renderWindow);

    // Set the parameters
    p_AgXProcessor.setFilmGamma(filmName,(float)gamma);

    // Call the base class process member, this will call the derived templated process code
    p_AgXProcessor.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

AgXEmulsionPluginFactory::AgXEmulsionPluginFactory()
    : OFX::PluginFactoryHelper<AgXEmulsionPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void AgXEmulsionPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    // Basic labels
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    // Add the supported contexts, only filter at the moment
    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);

    // Add supported pixel depths
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    // Set a few flags
    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // Setup CUDA render capability flags on non-Apple system
#ifndef __APPLE__
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(false);
#endif
}

void AgXEmulsionPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip only in the filter context
    // Create the mandated source clip
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    // Make some pages and to things in
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // Film stock choice
    ChoiceParamDescriptor* fchoice = p_Desc.defineChoiceParam("filmStock");
    fchoice->setLabel("Film Stock");
    fchoice->appendOption("Kodak Portra 400");
    fchoice->appendOption("Kodak Vision3 250D");
    fchoice->setDefault(0);
    page->addChild(*fchoice);

    // Gamma factor
    DoubleParamDescriptor* gparam = p_Desc.defineDoubleParam("gammaFactor");
    gparam->setLabel("Gamma Factor");
    gparam->setHint("Density curve gamma scaling");
    gparam->setDefault(1.0);
    gparam->setRange(0.5,2.0);
    gparam->setIncrement(0.01);
    gparam->setDisplayRange(0.5,2.0);
    page->addChild(*gparam);
}

ImageEffect* AgXEmulsionPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new AgXEmulsionPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static AgXEmulsionPluginFactory agxEmulsionPlugin;
    p_FactoryArray.push_back(&agxEmulsionPlugin);
} 