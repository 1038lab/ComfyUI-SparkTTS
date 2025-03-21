# ComfyUI-SparkTTS v1.1.0
# This custom node for ComfyUI provides functionality for text-to-speech synthesis using SparkTTS,
# including voice creation, voice cloning, and advanced voice cloning with control over pitch and speed.
#
# Models License Notice:
# - SparkTTS: Apache-2.0 License (https://huggingface.co/SparkAudio/Spark-TTS-0.5B)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-SparkTTS

import os
import torch
import numpy as np
import tempfile
from pathlib import Path
import folder_paths
from typing import Dict, Any, Tuple, Optional, List, Union
import re

from .AILab_SparkTTS_Core import SparkTTSCore

_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        print("Initializing SparkTTS model...")
        _model_instance = SparkTTSCore()
    return _model_instance

class SparkTTS_VoiceCreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "This is the SparkTTS voice creator node, you can enter text to synthesize. Currently we only support English and Chinese.", 
                         "placeholder": "Enter text to synthesize. Use double line breaks to separate paragraphs."}),
                "gender": (["female", "male"], {"default": "female"}),
                "pitch": (["very_low", "low", "moderate", "high", "very_high"], {"default": "moderate"}),
                "speed": (["very_low", "low", "moderate", "high", "very_high"], {"default": "moderate"}),
            },
            "optional": {
                "batch_texts": ("STRING", {"multiline": True, "default": "", 
                               "placeholder": "Enter additional texts here (one paragraph per line). Each paragraph will be processed separately and combined into a single audio output. Use this for better control over pacing and intonation of longer content. Format: one paragraph per line, blank lines are ignored."}),
            },
            "hidden": {
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"
    CATEGORY = "🧪AILab/🔊Audio"
    
    def synthesize(self, text, gender, pitch, speed, batch_texts: Optional[str] = None, 
                  temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95):
        model = get_model()
        
        all_texts = [text.strip()]
        if batch_texts and batch_texts.strip():
            all_texts.extend([t.strip() for t in batch_texts.split("\n") if t.strip()])
        
        all_texts = [t for t in all_texts if t.strip()]
        
        if not all_texts:
            empty_audio = {"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000}
            return (empty_audio,)
        
        audio_data = []
        
        for t in all_texts:
            with torch.no_grad():
                wav = model.inference(
                    t, gender=gender, pitch=pitch, speed=speed,
                    temperature=temperature, top_k=top_k, top_p=top_p
                )
                audio_data.append(wav)
        
        if not audio_data:
            empty_audio = {"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000}
            return (empty_audio,)
            
        combined_wav = np.concatenate(audio_data)
        audio_tensor = torch.from_numpy(combined_wav).unsqueeze(0).unsqueeze(0).float()
        combined_audio = {"waveform": audio_tensor, "sample_rate": model.sample_rate}
        
        return (combined_audio,)

class SparkTTS_VoiceClone:    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "This is the SparkTTS voice clone node, you can clone the voice from a reference audio. Enter reference text to improve voice cloning quality. Currently we only support English and Chinese.", 
                         "placeholder": "Enter text to synthesize with the cloned voice. Use double line breaks to separate paragraphs."}),
                "reference_audio": ("AUDIO", {"tooltip": "The audio sample to clone the voice from."}),
                "reference_text": ("STRING", {"multiline": True, "default": "", 
                                 "placeholder": "Enter the exact text spoken in the reference audio. This significantly improves voice cloning quality by helping the model understand the speaker's pronunciation patterns."}),
                "max_tokens": ("INT", {"default": 3000, "min": 500, "max": 5000, "step": 100, 
                             "tooltip": "Controls the maximum length of generated speech. Higher values allow for longer text but use more memory. Reduce this value if you encounter out-of-memory errors. Increase it for very long texts."}),
            },
            "optional": {
                "batch_texts": ("STRING", {"multiline": True, "default": "", 
                               "placeholder": "Enter additional texts here (one paragraph per line). Each paragraph will be processed separately with the cloned voice and combined into a single audio output. Use this for better control over pacing and intonation of longer content. Format: one paragraph per line, blank lines are ignored."}),
            },
            "hidden": {
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "🧪AILab/🔊Audio"
    
    def clone(self, text, reference_audio, reference_text, max_tokens,
              batch_texts: Optional[str] = None, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95):
        try:
            model = get_model()
            
            ref_text = reference_text.strip() if reference_text and reference_text.strip() else None
            
            waveform = reference_audio["waveform"].squeeze(0)
            sample_rate = reference_audio["sample_rate"]
            
            import torchaudio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_file_path = temp_file.name
            
            torchaudio.save(
                audio_file_path, 
                waveform, 
                sample_rate, 
                format="wav", 
                bits_per_sample=16, 
                encoding="PCM_S"
            )
            
            all_texts = [text.strip()]
            if batch_texts and batch_texts.strip():
                all_texts.extend([t.strip() for t in batch_texts.split("\n") if t.strip()])
            
            all_texts = [t for t in all_texts if t.strip()]
            
            if not all_texts:
                empty_audio = {"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000}
                return (empty_audio,)
            
            audio_data = []
            
            for t in all_texts:
                with torch.no_grad():
                    wav = model.inference(
                        t, prompt_speech_path=audio_file_path, prompt_text=ref_text,
                        max_new_tokens=max_tokens,
                        temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    audio_data.append(wav)
            
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            
            if not audio_data:
                empty_audio = {"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000}
                return (empty_audio,)
            
            combined_wav = np.concatenate(audio_data)
            audio_tensor = torch.from_numpy(combined_wav).unsqueeze(0).unsqueeze(0).float()
            combined_audio = {"waveform": audio_tensor, "sample_rate": model.sample_rate}
            
            return (combined_audio,)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            empty_audio = {"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000}
            return (empty_audio,)

class SparkTTS_AdvVoiceClone:    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "This is the SparkTTS advanced voice clone node, you can clone the voice from a reference audio with control over pitch and speed. Enter reference text to improve voice cloning quality. Currently we only support English and Chinese.", 
                         "placeholder": "Enter text to synthesize with the cloned voice. Use double line breaks to separate paragraphs."}),
                "reference_audio": ("AUDIO", {"tooltip": "The audio sample to clone the voice from."}),
                "reference_text": ("STRING", {"multiline": True, "default": "", 
                                 "placeholder": "Enter the exact text spoken in the reference audio. This significantly improves voice cloning quality by helping the model understand the speaker's pronunciation patterns."}),
                "pitch": (["very_low", "low", "moderate", "high", "very_high"], {"default": "moderate"}),
                "speed": (["very_low", "low", "moderate", "high", "very_high"], {"default": "moderate"}),
                "max_tokens": ("INT", {"default": 3000, "min": 500, "max": 5000, "step": 100, 
                             "tooltip": "Controls the maximum length of generated speech. Higher values allow for longer text but use more memory. Reduce this value if you encounter out-of-memory errors. Increase it for very long texts."}),
            },
            "optional": {
                "batch_texts": ("STRING", {"multiline": True, "default": "", 
                               "placeholder": "Enter additional texts here (one paragraph per line). Each paragraph will be processed separately with the cloned voice and combined into a single audio output. Use this for better control over pacing and intonation of longer content. Format: one paragraph per line, blank lines are ignored."}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "🧪AILab/🔊Audio"
    
    def clone(self, text, reference_audio, reference_text, pitch, speed, max_tokens,
              batch_texts: Optional[str] = None, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95):
        try:
            model = get_model()
            gender = "female"
            
            ref_text = reference_text.strip() if reference_text and reference_text.strip() else None
            
            waveform = reference_audio["waveform"].squeeze(0)
            sample_rate = reference_audio["sample_rate"]
            
            import torchaudio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_file_path = temp_file.name
            
            torchaudio.save(
                audio_file_path, 
                waveform, 
                sample_rate, 
                format="wav", 
                bits_per_sample=16, 
                encoding="PCM_S"
            )
            
            all_texts = [text.strip()]
            if batch_texts and batch_texts.strip():
                all_texts.extend([t.strip() for t in batch_texts.split("\n") if t.strip()])
            
            all_texts = [t for t in all_texts if t.strip()]
            
            if not all_texts:
                empty_audio = {"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000}
                return (empty_audio,)
            
            audio_data = []
            
            global_token_ids, _ = model.audio_tokenizer.tokenize(audio_file_path)
            
            for t in all_texts:
                with torch.no_grad():
                    control_prompt = model.process_prompt_control(gender, pitch, speed, t)
                    
                    global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])
                    
                    end_style_pos = control_prompt.find("<|end_style_label|>")
                    if end_style_pos != -1:
                        modified_prompt = (
                            control_prompt[:end_style_pos + len("<|end_style_label|>")] + 
                            "<|start_global_token|>" + 
                            global_tokens + 
                            "<|end_global_token|>" + 
                            control_prompt[end_style_pos + len("<|end_style_label|>"):]
                        )
                    else:
                        modified_prompt = control_prompt + "<|start_global_token|>" + global_tokens + "<|end_global_token|>"
                    
                    model_inputs = model.tokenizer([modified_prompt], return_tensors="pt").to(model.device)
                    
                    generated_ids = model.model.generate(
                        **model_inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    )
                    
                    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
                    predicts = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    semantic_tokens = re.findall(r"bicodec_semantic_(\d+)", predicts)
                    pred_semantic_ids = torch.tensor([int(token) for token in semantic_tokens]).long().unsqueeze(0) if semantic_tokens else torch.zeros(1, 0).long()
                    
                    wav = model.audio_tokenizer.detokenize(global_token_ids.to(model.device).squeeze(0), pred_semantic_ids.to(model.device))
                    
                    if isinstance(wav, torch.Tensor):
                        wav = wav.cpu().numpy()
                    
                    audio_data.append(wav)
            
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            
            if not audio_data:
                empty_audio = {"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000}
                return (empty_audio,)
            
            combined_wav = np.concatenate(audio_data)
            audio_tensor = torch.from_numpy(combined_wav).unsqueeze(0).unsqueeze(0).float()
            combined_audio = {"waveform": audio_tensor, "sample_rate": model.sample_rate}
            
            return (combined_audio,)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            empty_audio = {"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000}
            return (empty_audio,)

NODE_CLASS_MAPPINGS = {
    "SparkTTS_VoiceCreator": SparkTTS_VoiceCreator,
    "SparkTTS_VoiceClone": SparkTTS_VoiceClone,
    "SparkTTS_AdvVoiceClone": SparkTTS_AdvVoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SparkTTS_VoiceCreator": "SparkTTS Voice Creator",
    "SparkTTS_VoiceClone": "SparkTTS Voice Clone",
    "SparkTTS_AdvVoiceClone": "SparkTTS Advanced Voice Clone",
} 