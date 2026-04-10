# 🇺🇸🇨🇳 Bilingual Voice Guide (English + Chinese)

## 概述 Overview

Z-BOT现在支持双语模式！GLM会生成英文和中文两种语言的响应，speaker会依次播放两种语言。

Z-BOT now supports bilingual mode! GLM generates responses in both English and Chinese, and the speaker plays both languages sequentially.

---

## 🚀 快速开始 Quick Start

### 1. 运行统一检测系统 Run Unified Detection

**终端 1 Terminal 1**:
```bash
cd "c:/WallNet Detection/zbot-eyes"
python -m net_inspector.unified_runner --camera 0
```

### 2. 启动双语语音播报 Start Bilingual Voice

**终端 2 Terminal 2**:
```bash
python speaker_anime_voice.py
```

默认配置 Default settings:
- ✅ 双语模式开启 Bilingual mode ON
- 🇺🇸 英文声音: Aria (年轻美国女孩) English: Aria (young American girl)
- 🇨🇳 中文声音: Xiaoxiao (可爱中国女孩) Chinese: Xiaoxiao (cute Chinese girl)

---

## 📊 系统流程 System Flow

```
Detection → GLM API (logs + image)
                ↓
    GLM Response (bilingual):
    "Hello Judges! I detected..."
    ---
    "各位评委好！我检测到..."
                ↓
    GLMCurrentResponse.txt
                ↓
    Speaker (bilingual mode)
                ↓
    🇺🇸 Aria speaks English
    (0.5s pause)
    🇨🇳 Xiaoxiao speaks Chinese
                ↓
    🔊 Judges hear both!
```

---

## 🎯 示例响应 Example Response

### GLM生成 GLM generates:

```
Hello Judges! I'm Z-BOT, your wall-climbing inspection robot. I detected debris with 100% confidence across all readings. Additionally, there's partial net coverage ranging from 7.1% to 7.65%.
---
各位评委好！我是Z-BOT，您的爬墙检测机器人。我以100%的置信度检测到碎片。此外，安全网部分覆盖范围为7.1%至7.65%。
```

### Speaker播放 Speaker plays:

1. 🇺🇸 **Aria** (English): "Hello Judges! I'm Z-BOT..."
2. ⏸️ 0.5秒暂停 0.5s pause
3. 🇨🇳 **Xiaoxiao** (Chinese): "各位评委好！我是Z-BOT..."

---

## ⚙️ 配置选项 Configuration Options

### 仅英文模式 English Only Mode

```bash
python speaker_anime_voice.py --no-bilingual
```

### 更换英文声音 Change English Voice

```bash
# 中国女孩声音 Chinese girl voice
python speaker_anime_voice.py --voice chinese_cute

# 韩国女孩声音 Korean girl voice
python speaker_anime_voice.py --voice korean_cute

# 查看所有声音 List all voices
python speaker_anime_voice.py --list-voices
```

### 更改检查间隔 Change Check Interval

```bash
# 每1秒检查一次 Check every 1 second
python speaker_anime_voice.py --interval 1.0
```

---

## 🎨 声音选项 Voice Options

### 英文声音 English Voices:
- `english_young` → **Aria** (年轻美国女孩) ⭐ 默认 Default
- `english_mature` → Jenny (成熟美国女性)
- `english_british` → Sonia (英国女性)

### 中文声音 Chinese Voices (自动用于中文部分 Auto-used for Chinese):
- **Xiaoxiao** (可爱中国女孩) ⭐ 固定 Fixed
- Xiaoyi (温暖中国女孩)

### 其他亚洲声音 Other Asian Voices:
- `japanese_cute` → Nanami (日本女孩，口音重 Heavy accent)
- `korean_cute` → SunHi (韩国女孩)

---

## 🔧 故障排除 Troubleshooting

### 只听到英文 Only hearing English

检查GLM响应是否包含 "---" 分隔符：
Check if GLM response contains "---" separator:

```bash
cat GLMCurrentResponse.txt
```

应该看到 Should see:
```
English text...
---
中文文本...
```

### 中文发音不清楚 Chinese pronunciation unclear

中文部分固定使用Xiaoxiao声音，这是最清晰的中文女声。
Chinese part uses Xiaoxiao voice (fixed), which is the clearest Chinese female voice.

### 两种语言之间没有停顿 No pause between languages

默认有0.5秒停顿。如果需要更长停顿，编辑 `speaker_anime_voice.py`:
Default pause is 0.5s. For longer pause, edit `speaker_anime_voice.py`:

```python
time.sleep(0.5)  # Change to 1.0 or 2.0
```

---

## 📝 测试 Testing

### 测试英文声音 Test English Voice

```bash
python speaker_anime_voice.py --test "Hello Judges! I am Z-BOT!"
```

### 测试双语 Test Bilingual

创建测试文件 Create test file:
```bash
echo "Hello Judges! I am Z-BOT!
---
各位评委好！我是Z-BOT！" > test_bilingual.txt
```

然后运行 Then run:
```bash
python speaker_anime_voice.py --file test_bilingual.txt
```

---

## 🎉 Challenge Cup演示建议 Challenge Cup Demo Tips

### 场景1：最大影响 Maximum Impact
- ✅ 使用双语模式 Use bilingual mode
- ✅ 英文：Aria (年轻、清晰) English: Aria (young, clear)
- ✅ 中文：Xiaoxiao (可爱、自然) Chinese: Xiaoxiao (cute, natural)
- ✅ 评委听到两种语言，印象深刻！Judges hear both, very impressive!

### 场景2：专业展示 Professional Demo
- ✅ 双语模式确保所有评委都能理解 Bilingual ensures all judges understand
- ✅ 中国评委听中文，国际评委听英文 Chinese judges hear Chinese, international judges hear English
- ✅ 展示国际化能力 Shows internationalization capability

---

## ✅ 总结 Summary

1. ✅ GLM自动生成双语响应 GLM auto-generates bilingual responses
2. ✅ Speaker依次播放英文和中文 Speaker plays English then Chinese
3. ✅ 英文：Aria (年轻美国女孩) English: Aria (young American girl)
4. ✅ 中文：Xiaoxiao (可爱中国女孩) Chinese: Xiaoxiao (cute Chinese girl)
5. ✅ 可选择仅英文模式 Optional English-only mode
6. ✅ 完美适配Challenge Cup！Perfect for Challenge Cup!

---

## 🚀 立即开始 Start Now

```bash
# 终端1: 运行检测 Terminal 1: Run detection
python -m net_inspector.unified_runner --camera 0

# 终端2: 启动双语语音 Terminal 2: Start bilingual voice
python speaker_anime_voice.py
```

现在你的机器人会用两种语言说话了！🎉
Now your robot speaks in two languages! 🎉
