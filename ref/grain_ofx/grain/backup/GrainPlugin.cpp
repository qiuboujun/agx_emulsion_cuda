#include "GrainPlugin.h"

#include <stdio.h>
#include <random>
#include <iostream>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

#define kPluginName "Grain"
#define kPluginGrouping "OpenFX JQ"
#define kPluginDescription "Apply grain to RGB channels"
#define kPluginIdentifier "com.JQ.Grain"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

class GrainProcessor : public OFX::ImageProcessor
{
public:
    explicit GrainProcessor(OFX::ImageEffect& p_Instance);

    virtual void processImagesCUDA();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setParams(float p_muR, float p_sigmaR, float p_NmonteCarlo, float p_sigmaFilter, float p_blend);
    void setFrameTime(double frameTime) {_frameTime = frameTime; }
private:
    OFX::Image* _srcImg;
    float _muR;
    float _sigmaR;
    float _NmonteCarlo;
    float _sigmaFilter;
    float _blend;
    double _frameTime;
};

GrainProcessor::GrainProcessor(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(int p_Width, int p_Height, float p_muR, float p_sigmaR, float p_NmonteCarlo, float p_sigmaFilter, int p_seed, float p_blend, float* p_Input, float* p_Output, int p_rnd);

void GrainProcessor::processImagesCUDA()
{

    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    // Generate a random number
    int frameSeed = static_cast<int>(_frameTime * 1000);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(1, 100);

    int randomValue = distr(gen);

    RunCudaKernel(width, height, _muR, _sigmaR, _NmonteCarlo, _sigmaFilter, frameSeed, _blend, input, output, randomValue);
}

void GrainProcessor::multiThreadProcessImages(OfxRectI p_ProcWindow)
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

void GrainProcessor::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void GrainProcessor::setParams(float p_muR, float p_sigmaR, float p_NmonteCarlo, float p_sigmaFilter, float p_blend)
{
    _muR = p_muR;
    _sigmaR = p_sigmaR;
    _NmonteCarlo = p_NmonteCarlo;
    _sigmaFilter = p_sigmaFilter;
    _blend = p_blend;
}


////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class GrainPlugin : public OFX::ImageEffect
{
public:
    explicit GrainPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Set up and run a processor */
    void setupAndProcess(GrainProcessor &p_GrainProcessor, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::DoubleParam* m_muR;
    OFX::DoubleParam* m_sigmaR;
    OFX::DoubleParam* m_NmonteCarlo;
    OFX::DoubleParam* m_sigmaFilter;
    OFX::DoubleParam* m_blend;
};

GrainPlugin::GrainPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    m_muR = fetchDoubleParam("grainSize");
    m_sigmaR = fetchDoubleParam("grainSigma");
    m_NmonteCarlo = fetchDoubleParam("grainQuality");
    m_sigmaFilter = fetchDoubleParam("grainSoft");
    m_blend = fetchDoubleParam("Blend");
}

void GrainPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        GrainProcessor GrainProcessor(*this);
        GrainProcessor.setFrameTime(p_Args.time);
        setupAndProcess(GrainProcessor, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool GrainPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    if (m_SrcClip)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }
    return false;
}

void GrainPlugin::setupAndProcess(GrainProcessor& p_GrainProcessor, const OFX::RenderArguments& p_Args)
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

    double gradius = 0.025, gsigma = 0.00, gmonte = 800.0, gfilter = 0.8, gblend = 1.0;

    gradius = m_muR->getValueAtTime(p_Args.time);
    gsigma = m_sigmaR->getValueAtTime(p_Args.time);
    gmonte = m_NmonteCarlo->getValueAtTime(p_Args.time);
    gfilter = m_sigmaFilter->getValueAtTime(p_Args.time);
    gblend = m_blend->getValueAtTime(p_Args.time);
    
    // Set the images
    p_GrainProcessor.setDstImg(dst.get());
    p_GrainProcessor.setSrcImg(src.get());
  
    // Setup OpenCL and CUDA Render arguments
    p_GrainProcessor.setGPURenderArgs(p_Args);

    // Set the render window
    p_GrainProcessor.setRenderWindow(p_Args.renderWindow);

    // Set the parameters
    p_GrainProcessor.setParams(gradius, gsigma, gmonte, gfilter, gblend);

    // Call the base class process member, this will call the derived templated process code
    p_GrainProcessor.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

GrainPluginFactory::GrainPluginFactory()
    : OFX::PluginFactoryHelper<GrainPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void GrainPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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

void GrainPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

    // Make overall scale params
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam("grainSize");
    param->setLabel("Grain Size");
    param->setHint("Adjust grain size");
    param->setDefault(0.025);
    param->setRange(0.001, 0.50);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 0.50);
    page->addChild(*param);

    param = p_Desc.defineDoubleParam("grainSigma");
    param->setLabel("Grain Variation");
    param->setHint("Make grain size slightly different");
    param->setDefault(0.000);
    param->setRange(0.000, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("grainQuality");
    param->setLabel("Grain Quality");
    param->setHint("Render quality of the grain");
    param->setDefault(800.0);
    param->setRange(100.0, 3000.0);
    param->setIncrement(50.0);
    param->setDisplayRange(100.0, 3000.0);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("grainSoft");
    param->setLabel("Grain Softness");
    param->setHint("Adjust how soft the grain looks");
    param->setDefault(0.8);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Blend");
    param->setLabel("Global Blend");
    param->setHint("blend effects");
    param->setDefault(1.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
}

ImageEffect* GrainPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new GrainPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static GrainPluginFactory grainPlugin;
    p_FactoryArray.push_back(&grainPlugin);
}
