#pragma once
#include "ofxsImageEffect.h"
#include <vector>
#include <string>

#define kAgXPluginName "AgX Emulsion"
#define kAgXPluginGrouping "AgX"
#define kAgXPluginDescription "Applies a film spectral simulation based on the AgX Emulsion algorithm."
#define kAgXPluginIdentifier "com.yourcompany.AgXEmulsionOFX"
#define kAgXPluginVersionMajor 1
#define kAgXPluginVersionMinor 0

namespace AgXEmu {

// Forward declarations
class AgXProcessor;

class AgXPlugin : public OFX::ImageEffect {
public:
    explicit AgXPlugin(OfxImageEffectHandle handle);

    virtual void render(const OFX::RenderArguments& args) override;
    virtual bool isIdentity(const OFX::IsIdentityArguments& args,
                            OFX::Clip*& identityClip,
                            double& identityTime) override;
private:
    void setupAndProcess(AgXProcessor& proc, const OFX::RenderArguments& args);

    // Clips
    OFX::Clip* _dstClip;
    OFX::Clip* _srcClip;

    // Parameters (minimal subset)
    OFX::ChoiceParam* _filmStock;
    OFX::DoubleParam* _exposureCompEV;
    OFX::BooleanParam* _autoExposure;

    OFX::ChoiceParam* _printPaper;
    OFX::DoubleParam* _printExposure;

    OFX::BooleanParam* _halationActive;
    OFX::RGBParam* _halationStrength; // percentage RGB
};

class AgXPluginFactory : public OFX::PluginFactoryHelper<AgXPluginFactory> {
public:
    AgXPluginFactory():PluginFactoryHelper(kAgXPluginIdentifier, kAgXPluginVersionMajor, kAgXPluginVersionMinor){}
    virtual void describe(OFX::ImageEffectDescriptor& desc) override;
    virtual void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context) override;
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum /*context*/) override {
        return new AgXPlugin(handle);
    }
};

} // namespace AgXEmu 