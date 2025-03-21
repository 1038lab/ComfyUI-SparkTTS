# ComfyUI-SparkTTS

ComfyUI_SparkTTS is a custom ComfyUI node implementation of [SparkTTS](https://github.com/SparkAudio/Spark-TTS), an advanced text-to-speech system that harnesses the power of large language models (LLMs) to generate highly accurate and natural-sounding speech.

![SparkTTS_Nodes](https://github.com/user-attachments/assets/1b3c618f-3e02-4114-88a5-05acd4dbb207)

## News & Updates
- **2025/03/21**: Update ComfyUI-SparkTTS to **v1.1.0** ( [update.md](https://github.com/1038lab/ComfyUI-SparkTTS/blob/main/update.md#v200-20250321) )

  - Integrated internationalization (i18n) support for multiple languages.
  - Improved user interface for dynamic language switching.
  - Enhanced accessibility for non-English speaking users with fully translatable features.

## Features

ComfyUI-SparkTTS provides the following main functionalities:

1. **Voice Creation**: Create a customized voice by adjusting parameters like gender, pitch, and speed.
2. **Voice Cloning**: Clone a voice from a reference audio sample.
3. **Advanced Voice Cloning**: Clone a voice from a reference audio with control over pitch and speed.
4. **Audio Processing**: Load and process audio files.
5. **Audio Recording**: Directly record audio for voice cloning or processing.

## Installation

### Method 1. install on ComfyUI-Manager, search `Comfyui-SparkTTS` and install
install requirment.txt in the ComfyUI-SparkTTS folder
  ```bash
  ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
  ```

### Method 2. Clone this repository to your ComfyUI custom_nodes folder:
  ```bash
  cd ComfyUI/custom_nodes
  git clone https://github.com/1038lab/ComfyUI-SparkTTS
  ```
  install requirment.txt in the ComfyUI-SparkTTS folder
  ```bash
  ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
  ```

### Method 3: Install via Comfy CLI
  Ensure `pip install comfy-cli` is installed.
  Installing ComfyUI `comfy install` (if you don't have ComfyUI Installed)
  install the ComfyUI-SparkTTS, use the following command:
  ```bash
  comfy node registry-install Comfyui-Spark-TTS
  ```
  install requirment.txt in the ComfyUI-SparkTTS folder
  ```bash
  ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
  ```

### 4. Manually download the models:
- The model will be automatically downloaded to `ComfyUI/models/TTS/SparkTTS/` when first time using the custom node.
- Manually download the SparkTTS-2.0 model by visiting this [link](https://huggingface.co/SparkAudio/Spark-TTS-0.5B/tree/main), then download the files and place them in the `/ComfyUI/models/TTSSparkTTS/SparkTTS-2.0` folder.

## Nodes

### SparkTTS Voice Creator 🔊

This node allows you to create a customized voice by adjusting parameters.

**Inputs:**
- `text`: Text to synthesize.
- `gender`: Gender of the voice (female or male).
- `pitch`: Pitch level of the voice (very_low, low, moderate, high, very_high).
- `speed`: Speed level of the voice (very_low, low, moderate, high, very_high).
- `batch_texts` (optional): Additional texts for better control over pacing and intonation.

**Outputs:**
- `audio`: Generated audio with the customized voice.

### SparkTTS Voice Clone 🔊

This node allows you to clone a voice from a reference audio sample.

**Inputs:**
- `text`: Text to synthesize with the cloned voice.
- `reference_audio`: The audio sample to clone the voice from.
- `reference_text`: Transcript of the reference audio to improve cloning quality.
- `max_tokens`: Controls the maximum length of generated speech.
- `batch_texts` (optional): Additional texts for better control over pacing and intonation.

**Outputs:**
- `audio`: Generated audio with the cloned voice.

### SparkTTS Advanced Voice Clone 🔊

This node allows you to clone a voice from a reference audio with control over pitch and speed.

**Inputs:**
- `text`: Text to synthesize with the cloned voice.
- `reference_audio`: The audio sample to clone the voice from.
- `reference_text`: Transcript of the reference audio to improve cloning quality.
- `pitch`: Pitch level of the voice.
- `speed`: Speed level of the voice.
- `max_tokens`: Controls the maximum length of generated speech.
- `batch_texts` (optional): Additional texts for better control over pacing and intonation.

**Outputs:**
- `audio`: Generated audio with the cloned voice.

### Audio Recorder 🔊

This node allows you to directly record audio.

**Inputs:**
- `recording`: Set to True to start recording audio.
- `recording_duration`: Recording duration in seconds.
- `sample_rate`: Audio sample rate.
- `noise_threshold`: Noise reduction threshold.
- `smoothing_kernel_size`: Size of the kernel used for smoothing the audio signal.

**Outputs:**
- `audio`: Recorded audio data.

## Example Workflows

Check the `example_workflows` directory for example workflows.

## Supported Languages

SparkTTS currently supports the following languages:
- English
- Chinese

## License

GPL-3.0 License
