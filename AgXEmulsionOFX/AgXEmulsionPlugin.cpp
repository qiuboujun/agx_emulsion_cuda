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
#include "DiffusionHalationKernel.cuh"
#include "GrainKernel.cuh"
#include "PaperKernel.cuh"

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
    void setImages(OFX::Image* src, OFX::Image* dst) {
        _srcImg = src; 
        _dstImg = dst;
    }
    // void setParams(float p_invertStrength);
    void setFrameTime(double frameTime) {_frameTime = frameTime; }
    void setFilmGamma(const char* film,float gamma){
        strncpy(_film, film, 63); 
        _film[63] = 0; 
        _gamma=gamma;
    }
    const char* getFilm() const {return _film;}
    float getGamma() const {return _gamma;}

    void setDiffusionHalation(float radius, float hal){ _radius = radius; _halStrength = hal; }
    void setGrain(float strength,unsigned int seed){ _grainStrength=strength; _grainSeed=seed; }
    void setExposure(float ev){ _exposureEV = ev; }
    void setPrintPaper(const char* paper){ strncpy(_paper,paper,63); _paper[63]=0; }

private:
    OFX::Image* _srcImg;
    // float _unused;
    double _frameTime;
    char _film[64];
    char _paper[64];
    float _gamma;
    float _radius;
    float _halStrength;
    float _grainStrength;
    unsigned int _grainSeed;
    float _exposureEV;
};

AgXEmulsionProcessor::AgXEmulsionProcessor(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
    _film[0] = '\0';  // Initialize as empty string
    _radius = 0.f;
    _halStrength = 0.f;
    _grainStrength = 0.f;
    _grainSeed = 0;
    _exposureEV = 0.f;
    _paper[0]='\0';
}

extern "C" void LaunchEmulsionCUDA(float* img, int width, int height, float gamma, float exposureEV);

void AgXEmulsionProcessor::processImagesCUDA()
{
    static char sFilm[64]="";
    static float sGamma=-1.0f;
    bool lutOK=true;
    
    // Debug: Print current values
    printf("DEBUG: _film='%s', _gamma=%f\n", _film, _gamma);
    printf("DEBUG: sFilm='%s', sGamma=%f\n", sFilm, sGamma);
    
    if(_film[0]!='\0' && (strcmp(_film,sFilm)!=0 || _gamma!=sGamma)){
        printf("DEBUG: LUT needs update\n");
        static void* helper = nullptr;
        static bool (*loadFn)(const char*,float*,float*,float*,float*) = nullptr;
        if(!helper){
            helper = dlopen("libAgXLUT.so", RTLD_LAZY);
            printf("DEBUG: dlopen result = %p\n", helper);
            if(helper) {
                loadFn = (bool(*)(const char*,float*,float*,float*,float*))dlsym(helper,"loadFilmLUT");
                printf("DEBUG: dlsym result = %p\n", (void*)loadFn);
            } else {
                printf("DEBUG: dlopen failed: %s\n", dlerror());
            }
        }
        if(loadFn && helper){
            float logE[601], r[601], g[601], b[601];
            bool loaded = loadFn(_film,logE,r,g,b);
            printf("DEBUG: loadFn result = %s\n", loaded ? "SUCCESS" : "FAILED");
            if(loaded){
                UploadLUTCUDA(logE,r,g,b);
                strncpy(sFilm,_film,63); sFilm[63]=0;
                sGamma=_gamma;
                printf("DEBUG: LUT uploaded successfully\n");
            } else {
                lutOK=false;
            }
        } else {
            lutOK=false;
            printf("DEBUG: No helper or loadFn available\n");
        }
    } else {
        printf("DEBUG: Using cached LUT\n");
    }
    // Load print LUT if paper selected
    static char sPaper[64]="";
    if(_paper[0]!='\0' && strcmp(_paper,sPaper)!=0){
        static void* helperP=nullptr; static bool (*loadPrint)(const char*,float*,float*,float*,float*)=nullptr;
        if(!helperP){ helperP=dlopen("libAgXLUT.so",RTLD_LAZY); if(helperP) loadPrint=(bool(*)(const char*,float*,float*,float*,float*))dlsym(helperP,"loadPrintLUT"); }
        if(loadPrint){float logE[601],r[601],g[601],b[601]; if(loadPrint(_paper,logE,r,g,b)){ printf("DEBUG: Print LUT loaded ok\n"); UploadPaperLUTCUDA(logE,r,g,b);
            // load spectra
            static bool (*loadSpec)(const char*,float*,float*,float*,float*,int* )=nullptr;
            if(!loadSpec){ loadSpec = (bool(*)(const char*,float*,float*,float*,float*,int*))dlsym(helperP,"loadPaperSpectra"); }
            float cSpec[200],mSpec[200],ySpec[200],dmin[200]; int nSpec=0;
            if(loadSpec && loadSpec(_paper,cSpec,mSpec,ySpec,dmin,&nSpec)){
               UploadPaperSpectraCUDA(cSpec,mSpec,ySpec,dmin,nSpec);
               printf("DEBUG: Spectra uploaded N=%d\n",nSpec);
            } else {printf("DEBUG: Spectra load fail\n");}

            strncpy(sPaper,_paper,63); sPaper[63]=0;} else {printf("DEBUG: Print LUT load fail\n");}}
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
        LaunchEmulsionCUDA(dstPtr, width, height, _gamma, _exposureEV);
        // Apply diffusion + halation if enabled
        if(_radius > 0.1f || _halStrength > 1e-5f){
            printf("DEBUG: LaunchDiffusionHalation radius=%f hal=%f\n", _radius, _halStrength);
            LaunchDiffusionHalationCUDA(dstPtr, width, height, _radius, _halStrength);
        }

        if(_grainStrength > 1e-5f){
            printf("DEBUG: LaunchGrain strength=%f seed=%u\n", _grainStrength, _grainSeed);
            LaunchGrainCUDA(dstPtr, width, height, _grainStrength, _grainSeed);
        }

        if(_paper[0]!='\0'){
            printf("DEBUG: LaunchPaperCUDA using %s\n", _paper);
            LaunchPaperCUDA(dstPtr,width,height);
        }
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
    OFX::DoubleParam* m_exposureEV;
    OFX::ChoiceParam* m_printPaper;
    OFX::DoubleParam* m_diffusionRadius;
    OFX::DoubleParam* m_halationStrength;
    OFX::DoubleParam* m_grainStrength;
    OFX::IntParam*    m_grainSeed;
};

AgXEmulsionPlugin::AgXEmulsionPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    m_filmStock = fetchChoiceParam("filmStock");
    m_gammaFactor = fetchDoubleParam("gammaFactor");
    m_exposureEV = fetchDoubleParam("exposureEV");
    m_printPaper = fetchChoiceParam("printPaper");
    m_diffusionRadius = fetchDoubleParam("diffusionRadius");
    m_halationStrength = fetchDoubleParam("halationStrength");
    m_grainStrength   = fetchDoubleParam("grainStrength");
    m_grainSeed       = fetchIntParam("grainSeed");
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
    // Get the render window and the time from the render arguments
    const OfxTime time = p_Args.time;
    const OfxRectI& renderWindow = p_Args.renderWindow;

    // Retrieve any instance data associated with this effect
    OFX::Image* src = m_SrcClip->fetchImage(time);
    OFX::Image* dst = m_DstClip->fetchImage(time);

    // Set images and other arguments
    p_AgXProcessor.setImages(src, dst);

    // Get parameter values
    int filmIdx; 
    m_filmStock->getValueAtTime(p_Args.time, filmIdx);
    printf("DEBUG: Film index from param = %d\n", filmIdx);
    
    const char* filmName = filmIdx==0?"kodak_portra_400":"kodak_vision3_250d";
    printf("DEBUG: Film name = %s\n", filmName);
    
    double gamma = m_gammaFactor->getValueAtTime(p_Args.time);
    double exposure = m_exposureEV->getValueAtTime(p_Args.time);
    double radius = m_diffusionRadius->getValueAtTime(p_Args.time);
    double hal   = m_halationStrength->getValueAtTime(p_Args.time);
    double grain = m_grainStrength->getValueAtTime(p_Args.time);
    int    gseed = m_grainSeed->getValueAtTime(p_Args.time);
    int paperIdx; m_printPaper->getValueAtTime(p_Args.time,paperIdx);
    const char* paperMap[] = {"", "kodak_2383", "kodak_2393", "fujifilm_crystal_archive_typeii", "kodak_ektacolor_edge", "kodak_endura_premier", "kodak_portra_endura", "kodak_supra_endura", "kodak_ultra_endura"};
    const char* paperName = paperIdx<9?paperMap[paperIdx]:"";
    printf("DEBUG: PrintPaper idx=%d name=%s\n",paperIdx,paperName);
    printf("DEBUG: ExposureEV = %f Radius = %f Halation = %f\n", exposure, radius, hal);
    printf("DEBUG: Grain strength = %f seed=%d\n", grain, gseed);
 
    p_AgXProcessor.setFilmGamma(filmName,(float)gamma);
    p_AgXProcessor.setDiffusionHalation((float)radius,(float)hal);
    p_AgXProcessor.setExposure((float)exposure);
    p_AgXProcessor.setPrintPaper(paperName);
    p_AgXProcessor.setGrain((float)grain,(unsigned int)gseed);

    // Setup OpenCL and CUDA Render arguments
    p_AgXProcessor.setGPURenderArgs(p_Args);

    // Set the render window
    p_AgXProcessor.setRenderWindow(p_Args.renderWindow);

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

    // Exposure EV
    DoubleParamDescriptor* eparam = p_Desc.defineDoubleParam("exposureEV");
    eparam->setLabel("Exposure EV");
    eparam->setHint("Exposure adjustment in stops (log2)");
    eparam->setDefault(0.0);
    eparam->setRange(-4.0,4.0);
    eparam->setIncrement(0.1);
    eparam->setDisplayRange(-4.0,4.0);
    page->addChild(*eparam);

    // Gamma factor
    DoubleParamDescriptor* gparam = p_Desc.defineDoubleParam("gammaFactor");
    gparam->setLabel("Gamma Factor");
    gparam->setHint("Density curve gamma scaling");
    gparam->setDefault(1.0);
    gparam->setRange(0.5,2.0);
    gparam->setIncrement(0.01);
    gparam->setDisplayRange(0.5,2.0);
    page->addChild(*gparam);

    // Diffusion radius
    DoubleParamDescriptor* dparam = p_Desc.defineDoubleParam("diffusionRadius");
    dparam->setLabel("Diffusion Radius");
    dparam->setHint("Gaussian diffusion radius in pixels");
    dparam->setDefault(3.0);
    dparam->setRange(0.0,25.0);
    dparam->setIncrement(0.1);
    dparam->setDisplayRange(0.0,25.0);
    page->addChild(*dparam);

    // Halation strength
    DoubleParamDescriptor* hparam = p_Desc.defineDoubleParam("halationStrength");
    hparam->setLabel("Halation Strength");
    hparam->setHint("Amount of red halation to add");
    hparam->setDefault(0.2);
    hparam->setRange(0.0,1.0);
    hparam->setIncrement(0.01);
    hparam->setDisplayRange(0.0,1.0);
    page->addChild(*hparam);

    // Grain strength
    DoubleParamDescriptor* gs = p_Desc.defineDoubleParam("grainStrength");
    gs->setLabel("Grain Strength");
    gs->setHint("Amount of grain noise");
    gs->setDefault(0.1);
    gs->setRange(0.0,1.0);
    gs->setIncrement(0.01);
    gs->setDisplayRange(0.0,1.0);
    page->addChild(*gs);

    // Grain seed
    IntParamDescriptor* gseed = p_Desc.defineIntParam("grainSeed");
    gseed->setLabel("Grain Seed");
    gseed->setHint("Random seed for grain pattern");
    gseed->setDefault(0);
    gseed->setRange(0,100000);
    gseed->setDisplayRange(0,100000);
    page->addChild(*gseed);

    // Print paper choice
    ChoiceParamDescriptor* pchoice = p_Desc.defineChoiceParam("printPaper");
    pchoice->setLabel("Print Paper");
    pchoice->appendOption("None");
    pchoice->appendOption("Kodak 2383");
    pchoice->appendOption("Kodak 2393");
    pchoice->appendOption("Fujifilm Crystal Archive Type II");
    pchoice->appendOption("Kodak Ektacolor Edge");
    pchoice->appendOption("Kodak Endura Premier");
    pchoice->appendOption("Kodak Portra Endura");
    pchoice->appendOption("Kodak Supra Endura");
    pchoice->appendOption("Kodak Ultra Endura");
    pchoice->setDefault(1);
    page->addChild(*pchoice);
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