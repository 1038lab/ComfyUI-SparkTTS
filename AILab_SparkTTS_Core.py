# ComfyUI-SparkTTS v1.0.0
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
import re
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import folder_paths
import sys

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("huggingface_hub not available, automatic model download disabled")

# 添加当前目录到sys.path
node_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(node_dir)

try:
    from sparktts.utils.file import load_config
    from sparktts.models.audio_tokenizer import BiCodecTokenizer
    from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP
except ImportError:
    print("Failed to import from sparktts. Please make sure the sparktts folder exists.")
    raise

class SparkTTSCore:    
    MODEL_FILES = {
        "base": [
            "BiCodec/model.safetensors",
            "LLM/config.json",
            "LLM/model.safetensors",
            "LLM/special_tokens_map.json",
            "LLM/tokenizer_config.json",
            "LLM/tokenizer.json",
            "LLM/vocab.json",
            "LLM/merges.txt"
        ],
        "wav2vec2": [
            "wav2vec2-large-xlsr-53/config.json",
            "wav2vec2-large-xlsr-53/preprocessor_config.json",
            "wav2vec2-large-xlsr-53/model.safetensors"
        ]
    }
    
    def __init__(self, model_dir: Optional[Union[str, Path]] = None, device: Optional[torch.device] = None):
        cuda_available = torch.cuda.is_available()
        self.device = device if device is not None else torch.device("cuda" if cuda_available else "cpu")
        
        if self.device.type == "cuda":
            print(f"SparkTTS: Using CUDA device - {torch.cuda.get_device_name(0)}")
        else:
            print("SparkTTS: CUDA not available, using CPU for inference (this may be slow)")
            
        self._repo_id = "SparkAudio/Spark-TTS-0.5B"
        self._model_name = "Spark-TTS-0.5B"
        
        if model_dir is None:
            model_dir = self._get_default_model_path()
        
        self.model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        if HF_AVAILABLE:
            self._ensure_model_files()
            
        config_path = self.model_dir / "config.yaml"
        self.configs = load_config(str(config_path))
        self.sample_rate = self.configs["sample_rate"]
        
        self._initialize_inference()
        
    def _get_default_model_path(self) -> Path:
        model_path = Path(folder_paths.models_dir) / "TTS" / "SparkTTS" / self._model_name
        return model_path

    def _ensure_model_files(self) -> bool:
        if not HF_AVAILABLE:
            return False
            
        try:
            repo_files = list_repo_files(self._repo_id)
        except Exception as e:
            print(f"Failed to get repo files: {e}")
            return False
            
        all_files = self.MODEL_FILES["base"] + self.MODEL_FILES["wav2vec2"]
        missing_files = []
        
        for file in all_files:
            file_path = self.model_dir / file
            if not file_path.exists() and file in repo_files:
                missing_files.append(file)
                os.makedirs(file_path.parent, exist_ok=True)
        
        if missing_files:
            print(f"Downloading {len(missing_files)} missing model files...")
            
            for file in missing_files:
                try:
                    hf_hub_download(
                        repo_id=self._repo_id, 
                        filename=file, 
                        local_dir=str(self.model_dir),
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    if not "generation_config.json" in file:
                        print(f"Failed to download {file}: {e}")
                        return False
            
        return True

    def _initialize_inference(self):
        llm_path = self.model_dir / "LLM"
        self.tokenizer = AutoTokenizer.from_pretrained(str(llm_path))
        self.model = AutoModelForCausalLM.from_pretrained(str(llm_path))
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.model.to(self.device)

    def process_prompt(self, text: str, prompt_speech_path: Path, prompt_text: Optional[str] = None) -> Tuple[str, torch.Tensor]:
        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(prompt_speech_path)
        global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])

        if prompt_text is not None:
            semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()])
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        return "".join(inputs), global_token_ids

    def process_prompt_control(self, gender: str, pitch: str, speed: str, text: str) -> str:
        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribute_tokens = "".join([gender_tokens, pitch_label_tokens, speed_label_tokens])

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribute_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    @torch.no_grad()
    def inference(self, 
                 text: str, 
                 prompt_speech_path: Optional[Path] = None, 
                 prompt_text: Optional[str] = None, 
                 gender: Optional[str] = None, 
                 pitch: Optional[str] = None, 
                 speed: Optional[str] = None, 
                 temperature: float = 0.8, 
                 top_k: int = 50, 
                 top_p: float = 0.95, 
                 max_new_tokens: int = 1000) -> np.ndarray:
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)
        else:
            prompt, global_token_ids = self.process_prompt(text, prompt_speech_path, prompt_text)

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        semantic_tokens = re.findall(r"bicodec_semantic_(\d+)", predicts)
        pred_semantic_ids = torch.tensor([int(token) for token in semantic_tokens]).long().unsqueeze(0) if semantic_tokens else torch.zeros(1, 0).long()

        if gender is not None:
            global_tokens = re.findall(r"bicodec_global_(\d+)", predicts)
            global_token_ids = torch.tensor([int(token) for token in global_tokens]).long().unsqueeze(0).unsqueeze(0) if global_tokens else torch.zeros(1, 1, 0).long()

        wav = self.audio_tokenizer.detokenize(global_token_ids.to(self.device).squeeze(0), pred_semantic_ids.to(self.device))
        
        if isinstance(wav, torch.Tensor):
            return wav.cpu().numpy()
        return wav

    def synthesize(self, text: str, gender: str = "female", pitch: str = "moderate", speed: str = "moderate", **kwargs) -> np.ndarray:
        return self.inference(text, gender=gender, pitch=pitch, speed=speed, **kwargs)
    
    def clone(self, text: str, prompt_speech_path: Path, prompt_text: Optional[str] = None, **kwargs) -> np.ndarray:
        return self.inference(text, prompt_speech_path=prompt_speech_path, prompt_text=prompt_text, **kwargs)
    
    def batch_synthesize(self, texts: List[str], gender: str = "female", pitch: str = "moderate", speed: str = "moderate", **kwargs) -> np.ndarray:
        audio_data = []
        for text in texts:
            if text.strip():
                wav = self.synthesize(text, gender=gender, pitch=pitch, speed=speed, **kwargs)
                audio_data.append(wav)
                
        if not audio_data:
            return np.zeros(0, dtype=np.float32)
            
        audio_data_np = []
        for wav in audio_data:
            if isinstance(wav, torch.Tensor):
                audio_data_np.append(wav.cpu().numpy())
            else:
                audio_data_np.append(wav)
                
        combined_wav = np.concatenate(audio_data_np)
        return combined_wav
    
    def batch_clone(self, texts: List[str], prompt_speech_path: Path, prompt_text: Optional[str] = None, **kwargs) -> np.ndarray:
        audio_data = []
        for text in texts:
            if text.strip():
                wav = self.clone(text, prompt_speech_path=prompt_speech_path, prompt_text=prompt_text, **kwargs)
                audio_data.append(wav)
                
        if not audio_data:
            return np.zeros(0, dtype=np.float32)
            
        audio_data_np = []
        for wav in audio_data:
            if isinstance(wav, torch.Tensor):
                audio_data_np.append(wav.cpu().numpy())
            else:
                audio_data_np.append(wav)
                
        combined_wav = np.concatenate(audio_data_np)
        return combined_wav  