# ComfyUI-SparkTTS v1.0.0
# This custom node for ComfyUI provides functionality for audio recording.
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-SparkTTS

import numpy as np
import torch
import time
import librosa
import sounddevice as sd
from scipy import ndimage
from scipy.signal import lfilter
from comfy.utils import ProgressBar

class AudioRecorder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recording": ("BOOLEAN", {"default": False, "tooltip": "Set to True to start recording audio."}),
                "recording_duration": ("INT", {"default": 10, "min": 1, "max": 60, "step": 1, "tooltip": "Duration of the recording in seconds, between 1 and 60."}),
                "sample_rate": (["16000", "44100", "48000"], {"default": "48000", "tooltip": "Select the sample rate for audio recording."}),
                "noise_threshold": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 3.0, "step": 0.1, "tooltip": "Threshold for noise reduction, higher values reduce more noise."}),
                "smoothing_kernel_size": ("INT", {"default": 5, "min": 1, "max": 11, "step": 2, "tooltip": "Size of the kernel used for smoothing the audio signal."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_audio"
    CATEGORY = "🧪AILab/🔊Audio"

    def _compute_stft(self, audio, window_size):
        hop_length = window_size // 4
        return librosa.stft(audio, n_fft=window_size, hop_length=hop_length, win_length=window_size)

    def _compute_istft(self, spectrogram, window_size):
        hop_length = window_size // 4
        return librosa.istft(spectrogram, hop_length=hop_length, win_length=window_size)

    def _analyze_noise_profile(self, noise_sample, window_size):
        noise_spectrogram = self._compute_stft(noise_sample, window_size)
        return {
            'mean': np.mean(np.abs(noise_spectrogram), axis=1, keepdims=True),
            'std': np.std(np.abs(noise_spectrogram), axis=1, keepdims=True)
        }

    def _apply_spectral_gating(self, spectrogram, noise_profile, threshold):
        gate = noise_profile['mean'] + threshold * noise_profile['std']
        return np.where(np.abs(spectrogram) > gate, spectrogram, 0)

    def _apply_smoothing(self, mask, kernel_size):
        smoothed = ndimage.uniform_filter(mask, size=(kernel_size, kernel_size))
        return np.clip(smoothed * 1.2, 0, 1)

    def _remove_echo(self, audio, delay, decay):
        b = np.array([1.0, -decay])
        a = np.array([1.0, -decay * np.exp(-delay)])
        return lfilter(b, a, audio)

    def _record_audio(self, duration, sample_rate):
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        pb = ProgressBar(duration)
        for _ in range(duration * 2):
            time.sleep(0.5)
            pb.update(0.5)
        sd.wait()
        return audio.flatten()

    def _detect_noise_sample(self, audio, window_size):
        energy = librosa.feature.rms(y=audio, frame_length=window_size, hop_length=window_size//4)
        min_idx = np.argmin(energy)
        start = min_idx * (window_size//4)
        return audio[start:start + window_size*2]

    def process_audio(self, recording, recording_duration, noise_threshold, smoothing_kernel_size, sample_rate):
        if not recording:
            return (None,)

        try:
            sr = int(sample_rate)
            audio = self._record_audio(recording_duration, sr)

            audio = self._remove_echo(audio, delay=0.1, decay=0.5)

            noise_sample = self._detect_noise_sample(audio, 2048)
            noise_profile = self._analyze_noise_profile(noise_sample, 2048)

            spectrogram = self._compute_stft(audio, 2048)
            mask = np.ones_like(spectrogram)

            for _ in range(2):
                cleaned_spec = self._apply_spectral_gating(spectrogram, noise_profile, noise_threshold)
                mask = np.where(np.abs(cleaned_spec) > 0, 1, 0)
                mask = self._apply_smoothing(mask, smoothing_kernel_size//2+1)
                spectrogram = spectrogram * mask

            processed = self._compute_istft(spectrogram * mask, 2048)
            peak = np.max(np.abs(processed))
            processed = processed * (0.99 / peak) if peak > 0 else processed

            waveform = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
            final_audio = {"waveform": waveform, "sample_rate": sr}

        except Exception as e:
            print(f"Audio processing failed: {str(e)}")
            raise

        return (final_audio,)

NODE_CLASS_MAPPINGS = {
    "AudioRecorder": AudioRecorder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioRecorder": "Audio Recorder"
}