#pragma comment(lib, "dxgi")

#include <windows.h>
#include <d3dkmthk.h>
#include <dxgi.h>

extern "C" __declspec(dllexport) int* __cdecl HwSchEnabled() {
    int* returnValue = new int[3];
    IDXGIAdapter* pAdapter;
    IDXGIFactory* pFactory = NULL;

    NTSTATUS status;
    status = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory);
    if (status == 0) {
        // Find first valid DXGI adapter
        for (UINT i = 0; pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i)
            break;
        pFactory->Release();

        // Get LUID of said adapter
        DXGI_ADAPTER_DESC desc;
        pAdapter->GetDesc(&desc);
        D3DKMT_OPENADAPTERFROMLUID d3dAdapter;
        d3dAdapter.AdapterLuid = desc.AdapterLuid;

        // Open a D3DKMTAdapter handle
        status = D3DKMTOpenAdapterFromLuid(&d3dAdapter);
        
        if (status == 0) {
            // Get WDDM 2.7 info about said adapter
            D3DKMT_QUERYADAPTERINFO qai;
            D3DKMT_WDDM_2_7_CAPS caps;
            qai.hAdapter = d3dAdapter.hAdapter;
            qai.Type = KMTQAITYPE_WDDM_2_7_CAPS;
            qai.pPrivateDriverData = &caps;
            qai.PrivateDriverDataSize = sizeof(caps);
            status = D3DKMTQueryAdapterInfo(&qai);
            if (status == 0) {
                returnValue[0] = caps.HwSchSupported;
                returnValue[1] = caps.HwSchEnabled;
                returnValue[2] = caps.HwSchEnabledByDefault;
                return returnValue;
            }
        }
        D3DKMT_CLOSEADAPTER close;
        close.hAdapter = d3dAdapter.hAdapter;
        D3DKMTCloseAdapter(&close);
    }
    return returnValue;
}