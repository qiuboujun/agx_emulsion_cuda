#include "AgXEmulsionPlugin.h"
#include "ofxsCore.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"
#include <memory>
#include <iostream>

using namespace OFX;
namespace AgXEmu {

////////////////////////////////////////////////////////////////////////////////
// Processor stub – will eventually call CUDA kernels converted from agx_emulsion
////////////////////////////////////////////////////////////////////////////////
class AgXProcessor : public OFX::ImageProcessor {
public:
    explicit AgXProcessor(OFX::ImageEffect& instance):OFX::ImageProcessor(instance){}
    void setSrcImg(OFX::Image* src){_src=src;}
    void setParams(int filmStock,double expEV,bool autoExp,int printPaper,double printExp,bool halActive,OfxRGBColourD halStrength){
        _filmStock=filmStock;_expEV=expEV;_autoExp=autoExp;_printPaper=printPaper;_printExp=printExp;_halActive=halActive;_halStrength=halStrength;}
    virtual void multiThreadProcessImages(OfxRectI procWindow) override {
        // Just passthrough for now
        for(int y=procWindow.y1;y<procWindow.y2;++y){
            if(_effect.abort()) break;
            float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(procWindow.x1,y));
            for(int x=procWindow.x1;x<procWindow.x2;++x){
                float* srcPix = static_cast<float*>(_src? _src->getPixelAddress(x,y):nullptr);
                if(srcPix){for(int c=0;c<4;++c) dstPix[c]=srcPix[c];}
                else {for(int c=0;c<4;++c) dstPix[c]=0;}
                dstPix+=4;
            }
        }
    }
private:
    OFX::Image* _src=nullptr;
    int _filmStock;
    double _expEV;
    bool _autoExp;
    int _printPaper; double _printExp; bool _halActive; OfxRGBColourD _halStrength;
};

////////////////////////////////////////////////////////////////////////////////
// Plugin class implementation
////////////////////////////////////////////////////////////////////////////////
AgXPlugin::AgXPlugin(OfxImageEffectHandle handle):ImageEffect(handle){
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    _filmStock = fetchChoiceParam("filmStock");
    _exposureCompEV = fetchDoubleParam("exposureCompEV");
    _autoExposure = fetchBooleanParam("autoExposure");

    _printPaper = fetchChoiceParam("printPaper");
    _printExposure = fetchDoubleParam("printExposure");

    _halationActive = fetchBooleanParam("halationActive");
    _halationStrength = fetchRGBParam("halationStrength");
}

void AgXPlugin::render(const OFX::RenderArguments& args){
    AgXProcessor proc(*this);
    setupAndProcess(proc,args);
}

bool AgXPlugin::isIdentity(const OFX::IsIdentityArguments& args, Clip*& identityClip, double& identityTime){
    identityClip=_srcClip;
    identityTime=args.time;
    return true; // passthrough for now
}

void AgXPlugin::setupAndProcess(AgXProcessor& proc,const OFX::RenderArguments& args){
    std::auto_ptr<Image> dst(_dstClip->fetchImage(args.time));
    std::auto_ptr<Image> src(_srcClip->fetchImage(args.time));
    proc.setDstImg(dst.get());
    proc.setSrcImg(src.get());

    double hr,hg,hb; _halationStrength->getValueAtTime(args.time,hr,hg,hb);
    OfxRGBColourD halStrength{hr,hg,hb};
    int filmIdx; _filmStock->getValue(filmIdx);
    int paperIdx; _printPaper->getValue(paperIdx);
    proc.setParams(filmIdx, _exposureCompEV->getValue(), _autoExposure->getValue(), paperIdx, _printExposure->getValue(), _halationActive->getValue(), halStrength);

    OfxRectI renderWindow = dst->getBounds();
    proc.multiThreadProcessImages(renderWindow);
}

////////////////////////////////////////////////////////////////////////////////
// Factory describe functions
////////////////////////////////////////////////////////////////////////////////
void AgXPluginFactory::describe(ImageEffectDescriptor& desc){
    desc.setLabels(kAgXPluginName,kAgXPluginName,kAgXPluginName);
    desc.setPluginGrouping(kAgXPluginGrouping);
    desc.setPluginDescription(kAgXPluginDescription);
    desc.addSupportedContext(eContextFilter);
    desc.addSupportedBitDepth(eBitDepthFloat);
    desc.setSupportsTiles(false);
}

void AgXPluginFactory::describeInContext(ImageEffectDescriptor& desc, ContextEnum /*context*/){
    ClipDescriptor* srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);

    ClipDescriptor* dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);

    PageParamDescriptor* page = desc.definePageParam("Controls");

    // Film stock choice
    ChoiceParamDescriptor* choice = desc.defineChoiceParam("filmStock");
    choice->setLabel("Film Stock");
    choice->appendOption("Kodak Gold 200");
    choice->appendOption("Kodak Portra 400");
    choice->appendOption("Fujifilm C200");
    choice->setDefault(0);
    page->addChild(*choice);

    // Exposure compensation
    DoubleParamDescriptor* dparam = desc.defineDoubleParam("exposureCompEV");
    dparam->setLabel("Exposure Comp EV");
    dparam->setDefault(0.0); dparam->setRange(-10.0,10.0); dparam->setIncrement(0.1);
    page->addChild(*dparam);

    BooleanParamDescriptor* b = desc.defineBooleanParam("autoExposure");
    b->setLabel("Auto Exposure"); b->setDefault(true);
    page->addChild(*b);

    // Print paper choice
    ChoiceParamDescriptor* paper = desc.defineChoiceParam("printPaper");
    paper->setLabel("Print Paper");
    paper->appendOption("Kodak Supra Endura");
    paper->appendOption("Fujifilm Crystal Archive");
    paper->setDefault(0);
    page->addChild(*paper);

    dparam = desc.defineDoubleParam("printExposure");
    dparam->setLabel("Print Exposure"); dparam->setDefault(1.0); dparam->setRange(0.0,5.0); dparam->setIncrement(0.05);
    page->addChild(*dparam);

    // Halation
    b = desc.defineBooleanParam("halationActive");
    b->setLabel("Halation Active"); b->setDefault(true);
    page->addChild(*b);

    RGBParamDescriptor* rgb = desc.defineRGBParam("halationStrength");
    rgb->setLabel("Halation Strength %"); rgb->setDefault(3.0,0.3,0.1);
    rgb->setRange(0.0,0.0,0.0, 100.0,100.0,100.0);
    page->addChild(*rgb);
}

} // namespace AgXEmu 

// Register factory with host
namespace OFX {
    void Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray){
        static AgXEmu::AgXPluginFactory agxFactory;
        p_FactoryArray.push_back(&agxFactory);
    }
} // namespace OFX 