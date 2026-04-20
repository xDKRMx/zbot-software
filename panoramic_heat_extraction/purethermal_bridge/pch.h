#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <Ks.h>
#include <KsProxy.h>
#include <vidcap.h>
#include <ksmedia.h>
#include <atlbase.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <limits>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <bitset>
#include <chrono>

#pragma comment(lib, "mfplat")
#pragma comment(lib, "mf")
#pragma comment(lib, "mfreadwrite")
#pragma comment(lib, "mfuuid")
