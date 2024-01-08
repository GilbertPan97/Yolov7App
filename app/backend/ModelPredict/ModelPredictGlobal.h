#ifndef MODELPREDICTGLOBAL_H
#define MODELPREDICTGLOBAL_H

#ifdef WIN32
#include <windows.h>

#ifdef MP_SHARED_EXPORT
#   ifdef MP_SHARED_EXPORT
#       define MP_EXPORT __declspec(dllexport)
#   else
#       define MP_EXPORT __declspec(dllimport)
#   endif
#else
#   define MP_EXPORT    // used on windows, with static lib build
#endif

#else
#define MP_EXPORT       // used for linux or mac platform
#endif

#endif // MODELPREDICTGLOBAL_H