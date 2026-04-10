# 🎀 Anime Voice TTS Guide

## Kurulum

Gerekli paketler zaten yüklü:
```bash
pip install edge-tts pygame
```

## Kullanım

### 1. Temel Kullanım (Varsayılan Japon Anime Kızı Sesi)

```bash
cd "c:/WallNet Detection/zbot-eyes"
python speaker_anime_voice.py
```

Bu komut:
- `GLMCurrentResponse.txt` dosyasını izler
- Yeni response geldiğinde **Nanami** (Japon anime kızı) sesiyle okur
- 2 saniyede bir kontrol eder

### 2. Farklı Ses Seçenekleri

```bash
# Tüm mevcut sesleri listele
python speaker_anime_voice.py --list-voices
```

**Mevcut Anime Sesleri**:
- `japanese_cute` → Nanami (Sevimli Japon kız) ⭐ **ÖNERİLEN**
- `japanese_cheerful` → Aoi (Neşeli Japon kız)
- `chinese_cute` → Xiaoxiao (Sevimli Çinli kız)
- `chinese_warm` → Xiaoyi (Sıcak Çinli kız)
- `korean_cute` → SunHi (Sevimli Koreli kız)
- `english_cute` → Jenny (Sevimli Amerikalı kız)
- `english_cheerful` → Sonia (Neşeli İngiliz kız)

### 3. Özel Ses ile Çalıştırma

```bash
# Japon sevimli ses (varsayılan)
python speaker_anime_voice.py --voice japanese_cute

# Çinli sevimli ses
python speaker_anime_voice.py --voice chinese_cute

# İngiliz neşeli ses
python speaker_anime_voice.py --voice english_cheerful
```

### 4. Test Modu

Sesi test etmek için:
```bash
python speaker_anime_voice.py --test "Hello Judges! I am Z-BOT!"
```

### 5. Unified Runner ile Birlikte Kullanım

**Terminal 1**: Unified runner'ı çalıştır
```bash
python -m net_inspector.unified_runner --camera 0
```

**Terminal 2**: Anime voice speaker'ı çalıştır
```bash
python speaker_anime_voice.py --voice japanese_cute
```

Artık her GLM response geldiğinde anime kızı sesiyle duyacaksın! 🎀

## 🎯 Örnek Senaryo

1. **Unified runner** çalışıyor → Detection yapıyor
2. **GLM** response oluşturuyor:
   > "Hello Judges! I'm Z-BOT, your wall-climbing inspection robot. Based on my latest scan, I detected some debris with complete confidence (100%) across all readings."
3. **GLMCurrentResponse.txt** güncelleniyor
4. **Anime voice speaker** değişikliği algılıyor
5. **Nanami sesi** ile okuyor: 🎀
   > *"Herro Judges! I'm Z-BOT, your warr-crimbing inspection robot..."* (Japon aksanıyla)

## 🎨 Ses Özellikleri

### Japanese Cute (Nanami) - ÖNERİLEN ⭐
- **Ton**: Yüksek, sevimli
- **Hız**: Orta
- **Aksan**: Hafif Japon aksanı
- **Uygun**: Anime karakteri gibi, judges'ı eğlendirir

### Chinese Cute (Xiaoxiao)
- **Ton**: Yüksek, enerjik
- **Hız**: Hızlı
- **Aksan**: Hafif Çin aksanı
- **Uygun**: Neşeli, dinamik sunum

### English Cute (Jenny)
- **Ton**: Orta-yüksek, profesyonel
- **Hız**: Orta
- **Aksan**: Amerikan İngilizcesi
- **Uygun**: Daha profesyonel sunum istiyorsan

## 🔧 Gelişmiş Ayarlar

### Kontrol Aralığını Değiştirme
```bash
# 1 saniyede bir kontrol et (daha hızlı)
python speaker_anime_voice.py --interval 1.0

# 5 saniyede bir kontrol et (daha yavaş)
python speaker_anime_voice.py --interval 5.0
```

### Farklı Dosya İzleme
```bash
python speaker_anime_voice.py --file "custom_response.txt"
```

## 🎭 Challenge Cup İçin Öneriler

### Senaryo 1: Maksimum Eğlence (Önerilen)
```bash
python speaker_anime_voice.py --voice japanese_cute
```
- Judges güler ve eğlenir
- Anime kızı sesi dikkat çeker
- Unutulmaz sunum

### Senaryo 2: Profesyonel + Sevimli
```bash
python speaker_anime_voice.py --voice english_cute
```
- İngilizce daha net anlaşılır
- Yine de sevimli ton
- Profesyonel görünüm

### Senaryo 3: Enerjik Sunum
```bash
python speaker_anime_voice.py --voice chinese_cute
```
- Çok enerjik ve hızlı
- Dinamik sunum
- Dikkat çekici

## 🐛 Sorun Giderme

### Ses Çıkmıyor
```bash
# Ses kartını kontrol et
python -c "import pygame; pygame.mixer.init(); print('OK')"

# Test et
python speaker_anime_voice.py --test "Test message"
```

### Dosya Bulunamıyor
```bash
# Dosya yolunu kontrol et
python speaker_anime_voice.py --file "c:/WallNet Detection/GLMCurrentResponse.txt"
```

### Yavaş Çalışıyor
```bash
# Kontrol aralığını artır
python speaker_anime_voice.py --interval 3.0
```

## 📊 Sistem Akışı

```
Detection → GLM → GLMCurrentResponse.txt
                         ↓
                  Anime Voice Speaker
                         ↓
                  Edge TTS (Nanami)
                         ↓
                  temp_anime_voice.mp3
                         ↓
                  pygame.mixer
                         ↓
                  🔊 Speaker → Judges duyar!
                         ↓
                  😍 "Kawaii desu ne~!"
```

## 🎉 Özet

1. ✅ **Edge TTS** kullanıyor (Microsoft'un ücretsiz TTS servisi)
2. ✅ **7 farklı anime sesi** mevcut
3. ✅ **Otomatik monitoring** (GLMCurrentResponse.txt)
4. ✅ **Test modu** (sesi önce dene)
5. ✅ **Kolay kullanım** (tek komut)

**En iyi seçim**: `--voice japanese_cute` (Nanami) 🎀

Haydi bakalım, anime kızı sesiyle judges'ı şaşırt! ✨
