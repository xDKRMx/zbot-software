# 📥 Visual Studio Kurulumu ve DLL Derleme Rehberi

## 🎯 Özet (Summary)

FLIR Lepton 2.5 termal kamera için C++ DLL oluşturacağız. Bunun için Visual Studio gerekli.

---

## ADIM 1: Visual Studio İndirme (Download)

### 🔗 İndirme Linki (DOĞRUDAN LINK):

**Visual Studio 2022 Community (ÜCRETSİZ / FREE):**

```
https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false
```

**Alternatif Link:**
```
https://visualstudio.microsoft.com/downloads/
```
Sayfada "Community 2022" altındaki **"Free download"** butonuna tıkla.

---

## ADIM 2: Visual Studio Kurulumu (Installation)

### 2.1 İndirilen dosyayı çalıştır:
```
VisualStudioSetup.exe
```

### 2.2 Kurulum ekranında **"Desktop development with C++"** seçeneğini işaretle:

**ÖNEMLİ: Mutlaka bu seçenek seçilmeli!**

```
☑ Desktop development with C++
```

### 2.3 Sağ tarafta (Individual components) bu öğelerin seçili olduğundan emin ol:

```
☑ MSVC v143 - VS 2022 C++ x64/x86 build tools
☑ Windows 10 SDK (10.0.19041.0 veya daha yeni)
☑ C++ CMake tools for Windows
```

### 2.4 "Install" butonuna bas ve bekle:

**Kurulum süresi:** 20-30 dakika (internet hızına bağlı)
**Gerekli disk alanı:** ~7 GB

---

## ADIM 3: DLL Oluşturma (Build DLL)

### 3.1 Visual Studio kurulumunu doğrula:

```powershell
# MSBuild'in yüklü olduğunu kontrol et
where msbuild
```

**Beklenen çıktı:**
```
C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe
```

❌ **Eğer "could not find" hatası alırsan:**
Visual Studio düzgün kurulmamış. Tekrar kur.

---

### 3.2 DLL'i derle (Compile):

```powershell
cd "c:\WallNet Detection\panoramic_heat_extraction\purethermal_bridge"
.\build.bat
```

**Beklenen çıktı:**
```
========================================
Building PureThermal Bridge DLL
========================================

Found MSBuild: "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"

Building Release x64...
[Derleme çıktıları...]

========================================
BUILD SUCCESS!
========================================

DLL copied to: c:\WallNet Detection\panoramic_heat_extraction\PureThermalBridge.dll
```

---

## ADIM 4: DLL'i Test Et

```powershell
cd "c:\WallNet Detection\panoramic_heat_extraction"
python purethermal_python.py
```

**Başarılı çıktı:**
```
Testing Pure Thermal Camera...
[PureThermal] Connected: 160x120
Frame captured: (120, 160) dtype=uint16
✅ SUCCESS - Pure Thermal works!
```

---

## ❓ Sorun Giderme (Troubleshooting)

### Hata: "MSBuild not found"

**Çözüm:**
1. Visual Studio'yu KAPAT
2. Visual Studio Installer'ı aç
3. "Modify" butonuna bas
4. "Desktop development with C++" seçeneğini işaretle
5. "Modify" ile kurulumu güncelle

---

### Hata: "Windows SDK not found"

**Çözüm:**
1. Visual Studio Installer → Modify
2. "Individual components" sekmesi
3. Ara (Search): "Windows 10 SDK"
4. En az bir SDK versiyonu seç (10.0.19041.0 önerilen)
5. Modify

---

### Hata: "Cannot open source file 'Device.h'"

**Çözüm:**
```powershell
# Device.cpp ve Device.h dosyalarını kopyala
cd "c:\WallNet Detection\panoramic_heat_extraction"
copy "purethermal1-uvc-capture\mediafoundation\PureThermal\Device.cpp" "purethermal_bridge\"
copy "purethermal1-uvc-capture\mediafoundation\PureThermal\Device.h" "purethermal_bridge\"

# Tekrar derle
cd purethermal_bridge
.\build.bat
```

---

### Hata: "Pure Thermal device not found" (DLL derlenmiş ama cihaz bulunamıyor)

**Kontrol listesi:**
1. ✅ Pure Thermal USB'ye takılı mı?
2. ✅ Device Manager'da "Pure Thermal" görünüyor mu?
   - Win+X → Device Manager → Cameras
3. ✅ Başka bir program Pure Thermal'ı kullanıyor mu? (kapat)
4. ✅ Farklı USB porta dene

**Device Manager'da kontrol:**
```
Cameras
  └─ Pure Thermal (veya FLIR Lepton)
```

❌ **"Unknown Device" görünüyorsa:**
Pure Thermal sürücüleri kurulu değil. GroupGets'ten sürücü indir.

---

## 🎯 Özet Komutlar (Tüm Adımlar)

```powershell
# 1. Visual Studio kurulduktan sonra MSBuild'i kontrol et
where msbuild

# 2. DLL'i derle
cd "c:\WallNet Detection\panoramic_heat_extraction\purethermal_bridge"
.\build.bat

# 3. Test et
cd "c:\WallNet Detection\panoramic_heat_extraction"
python purethermal_python.py

# 4. GUI'yi çalıştır
python gui.py --thermal 0
```

---

## 📞 Yardım

**Visual Studio kurulum sorunları için:**
- Microsoft Docs: https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio

**Pure Thermal sürücü sorunları için:**
- GroupGets: https://groupgets.com/manufacturers/getlab/products/purethermal-2-flir-lepton-smart-i-o-module

---

## ✅ Başarı Kriterleri

Şu komutlar çalışırsa başarılısın:

1. ✅ `where msbuild` → MSBuild yolunu gösteriyor
2. ✅ `build.bat` → "BUILD SUCCESS!" mesajı
3. ✅ `python purethermal_python.py` → "SUCCESS - Pure Thermal works!"
4. ✅ `python gui.py --thermal 0` → Termal kamera açılıyor

**Bu adımları tamamladıktan sonra sergi için hazırsın!** 🚀
