from moviepy.editor import VideoFileClip
import tempfile
import shutil
import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from moviepy.video.io.VideoFileClip import VideoFileClip
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class VideoEditor():
    def __init__(self) -> None:
        # criar um objeto do tipo speech recognition
        self.recognizer = sr.Recognizer()
        self.folder_name = "audio-chunks"

    def convert_to_mono_wav(self, input_audio, output_audio):
        sound = AudioSegment.from_wav(input_audio)
        sound = sound.set_channels(1)  # Mono
        sound = sound.set_frame_rate(16000)  # 16 kHz
        sound.export(output_audio, format="wav")

    def extract_audio_video(self, video_path:str, model: str = 'google') -> str:
        # Carrega o arquivo de vídeo
        video_clip = VideoFileClip(video_path)
        # Extrai o áudio
        audio_clip = video_clip.audio
        if model == 'google':
            suffix = '.wav'
            codec = 'pcm_s16le'
        elif model == 'transformers':
            suffix = '.mp3'
            codec = 'mp3'
        # Cria um arquivo temporário para armazenar o áudio
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name
            audio_clip.write_audiofile(temp_audio_file_path, codec=codec, verbose=False)
        # Fecha os objetos de áudio e vídeo
        audio_clip.close()
        video_clip.close()
        # Move o arquivo temporário para a pasta do projeto
        shutil.move(temp_audio_file_path, f"{os.getcwd()}/audio{suffix}")
        if model == 'google':
            self.convert_to_mono_wav(f"{os.getcwd()}/audio.wav", "audio_file_converted.wav")
            os.remove(f"{os.getcwd()}/audio{suffix}")
            return f"{os.getcwd()}/audio_file_converted{suffix}"
        else:
            return f"{os.getcwd()}/audio{suffix}"

    def transcribe_audio_with_google(self, path):
        # usa o arquivo de áudio como fonte de áudio
        with sr.AudioFile(path) as source:
            audio_listened = self.recognizer.record(source)
            # try converting it to text
            text = self.recognizer.recognize_google(audio_listened, language="pt-BR")
        return text

    def transcribe_audio_with_transformers(self, audio_path: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        # return_timestamps definido como True para o caso de aúdios acima de 30 segundos
        result = pipe(audio_path, return_timestamps=True)
        text = result["text"]
        return text

    def get_large_audio_transcription_on_silence(self, path: str, phrase: str, model: str='google'):
        """EM CONSTRUÇÃO"""
        """Dividir o arquivo de áudio grande em pedaços e aplicar reconhecimento de fala em cada um desses pedaços"""
        # abre o arquivo de áudio usando o pydub
        sound = AudioSegment.from_file(path)
        data = {}
        # som de áudio dividido onde o silêncio é de 500 milissegundos ou mais e obtém pedaços
        chunks = split_on_silence(sound,
            min_silence_len = 500,
            silence_thresh = sound.dBFS-14,
            keep_silence=500,
        )
        # crie um diretório para armazenar os pedaços de áudio
        if not os.path.isdir(self.folder_name):
            os.mkdir(self.folder_name)
        # processar cada pedaço 
        for i, audio_chunk in enumerate(chunks, start=1):
            # exporte o pedaço de áudio e salve-o no diretório `folder_name`.
            chunk_filename = os.path.join(self.folder_name, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            # transcrever o pedaço
            try:
                if model == 'google':
                    text = self.transcribe_audio_with_google(chunk_filename)
                elif model == 'transformers':
                    text = self.transcribe_audio_with_transformers(chunk_filename)
                else:
                    Exception('Escolha um modelo válido')
            except sr.UnknownValueError as e:
                continue
            else:
                if phrase.lower() in text.lower():
                    position = text.find(phrase)
                    if position <= 14: # 20% do pedaço do áudio
                        data = {"Second_initial":((i-1) * 5) + 1, "Second_end":(i * 5), "Second_cut":((i-1) * 5) + 1}
                    elif position > 14 and position <= 28: # 40% do pedaço do áudio
                        data = {"Second_initial":((i-1) * 5) + 1, "Second_end":(i * 5), "Second_cut":((i-1) * 5) + 2}
                    elif position > 28 and position <= 42: # 60% do pedaço do áudio
                        data = {"Second_initial":((i-1) * 5) + 1, "Second_end":(i * 5), "Second_cut":((i-1) * 5) + 3}
                    elif position > 42 and position <= 56: # 80% do pedaço do áudio
                        data = {"Second_initial":((i-1) * 5) + 1, "Second_end":(i * 5), "Second_cut":((i-1) * 5) + 4}
                    else: # 100% do pedaço do áudio
                        data = {"Second_initial":((i-1) * 5) + 1, "Second_end":(i * 5), "Second_cut":((i-1) * 5) + 5}
                    break
        # return the text for all chunks detected
        return data

    def get_large_audio_transcription_fixed_interval(self, path: str, phrase: str, model: str = 'google'):
        """EM CONSTRUÇÂO, O TEMPO ESTÁ FIXO EM 5 SEGUNDOS DE ÁUDIO"""
        """Dividir o arquivo de áudio grande em blocos de intervalo fixo e aplicar reconhecimento de fala em cada um desses blocos"""
        data = {}
        # abre o arquivo de áudio usando o pydub
        sound = AudioSegment.from_file(path)  
        # split the audio file into chunks
        chunk_length_ms = int(1000 * 60 * (1/12)) # converte os 5 segundos de aúdio para millisegundos
        chunks = [sound[i:i + chunk_length_ms] for i in range(0, len(sound), chunk_length_ms)]
        # criar diretório para salvar os pedaços do áudio
        if not os.path.isdir(self.folder_name):
            os.mkdir(self.folder_name)
        # processar cada pedaço 
        for i, audio_chunk in enumerate(chunks, start=1):
            # exporta o pedaço do áudio e sava-ló no diretório.
            chunk_filename = os.path.join(self.folder_name, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            # transcrever o áudio do pedaço cortado
            try:
                if model == 'google':
                    text = self.transcribe_audio_with_google(chunk_filename)
                elif model == 'transformers':
                    text = self.transcribe_audio_with_transformers(chunk_filename)
                else:
                    Exception('Escolha um modelo válido')
            except sr.UnknownValueError as e:
                continue
            else:
                if phrase.lower() in text.lower():
                    position = text.find(phrase)
                    if position <= 14: # 20% do pedaço do áudio
                        data = {"Second_initial":((i-1) * 5) + 1, "Second_end":(i * 5), "Second_cut":((i-1) * 5) + 1}
                    elif position > 14 and position <= 28: # 40% do pedaço do áudio
                        data = {"Second_initial":((i-1) * 5) + 1, "Second_end":(i * 5), "Second_cut":((i-1) * 5) + 2}
                    elif position > 28 and position <= 42: # 60% do pedaço do áudio
                        data = {"Second_initial":((i-1) * 5) + 1, "Second_end":(i * 5), "Second_cut":((i-1) * 5) + 3}
                    elif position > 42 and position <= 56: # 80% do pedaço do áudio
                        data = {"Second_initial":((i-1) * 5) + 1, "Second_end":(i * 5), "Second_cut":((i-1) * 5) + 4}
                    else: # 100% do pedaço do áudio
                        data = {"Second_initial":((i-1) * 5) + 1, "Second_end":(i * 5), "Second_cut":((i-1) * 5) + 5}
                    break
        # retorna os dados com as informações para o corte do vídeo
        return data

    def cut_video_by_sentence(self, path:str, phrase:str, model: str = 'google', until: int = 0):
        # Carregar o vídeo
        video = VideoFileClip(rf"{path}")
        # Definir o tempo de início e fim do corte (em segundos)
        audio_file = self.extract_audio_video(rf"{path}", model=model)
        video_cut_infor = self.get_large_audio_transcription_fixed_interval(audio_file, phrase, model=model)
        os.remove(audio_file)
        shutil.rmtree(self.folder_name)
        # Cortar o vídeo
        if until == 0:
            video_cortado = video.subclip(video_cut_infor['Second_cut'])
        else:
            video_cortado = video.subclip(video_cut_infor['Second_cut'], until)
        # Salvar o vídeo cortado
        video_cortado.write_videofile(rf"{os.getcwd()}/video_cortado.mp4", codec="libx264")

    def cut_video_by_timestamps(self, path:str, initial: int, until: int = 0):
        # Carregar o vídeo
        video = VideoFileClip(rf"{path}")
        video_cortado = video.subclip(19, 111)
        if until == 0:
            video_cortado = video.subclip(initial)
        else:
            video_cortado = video.subclip(initial, until)
        # Salvar o vídeo cortado
        video_cortado.write_videofile(rf"{os.getcwd()}/video_cortado.mp4", codec="libx264")

phrase = "hoje vamos falar"
path = "B:\VideoEditor\WhatsApp Video 2024-10-10 at 10.05.40_e3c898dd.mp4"
VEditor = VideoEditor()
VEditor.cut_video_by_sentence(path, phrase, model='transformers')
