#include <dlfcn.h>
#include <iostream>

extern "C" {
    int OfxGetNumberOfPlugins();
    void* OfxGetPlugin(int nth);
}

int main() {
    void* handle = dlopen("./AgXEmulsionPlugin.ofx", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot load plugin: " << dlerror() << std::endl;
        return 1;
    }

    // Get function pointers
    int (*getNumPlugins)() = (int(*)())dlsym(handle, "OfxGetNumberOfPlugins");
    void* (*getPlugin)(int) = (void*(*)(int))dlsym(handle, "OfxGetPlugin");

    if (!getNumPlugins || !getPlugin) {
        std::cerr << "Cannot find required functions" << std::endl;
        dlclose(handle);
        return 1;
    }

    int numPlugins = getNumPlugins();
    std::cout << "Number of plugins: " << numPlugins << std::endl;

    for (int i = 0; i < numPlugins; ++i) {
        void* plugin = getPlugin(i);
        std::cout << "Plugin " << i << ": " << plugin << std::endl;
    }

    dlclose(handle);
    return 0;
} 