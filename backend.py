import os

import argparse
from argparse import Namespace

# import logging
import tempfile
from os.path import join as opj
import re

import torch
import torchaudio
import numpy as np

import whisper

from pyannote.audio import Pipeline

from denoiser.audio import Audioset
from denoiser import distrib, pretrained
from denoiser.audio import Audioset, find_audio_files

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import AudioFileClip, concatenate_audioclips


# logger = logging.getLogger(__name__)

# def add_flags(parser):
#     """
#     Add the flags for the argument parser that are related to model loading and evaluation"
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = "cpu"
#     pretrained.add_model_flags(parser)
#     parser.add_argument('--device', default=device)
#     parser.add_argument('--num_workers', type=int, default=0)
#     parser.add_argument('--streaming', action="store_true",
#                         help="true streaming evaluation for Demucs")


# parser = argparse.ArgumentParser(
#         'denoiser.enhance',
#         description="Speech enhancement using Demucs - Generate enhanced files")
# add_flags(parser)
# parser.add_argument("--batch_size", default=1, type=int, help="batch size")
# parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
#                     default=logging.INFO, help="more loggging")

# group = parser.add_mutually_exclusive_group()
# group.add_argument("--noisy_dir", type=str, default=None,
#                    help="directory including noisy wav files")
# group.add_argument("--noisy_json", type=str, default=None,
#                    help="json file including noisy wav files")

# args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# args = Namespace(model_path=None,
#                  dns48=False,
#                  dns64=False,
#                  master64=False,
#                  valentini_nc=False,
#                 batch_size=1,
#                  noisy_dir=None,
#                  noisy_json=None,
#                  streaming=False,
#                  verbose=1,
#                  device=device,
#                  num_workers=2
#                  )



denoise_model = pretrained.get_model(Namespace(model_path=None, dns48=False, dns64=False, master64=False, valentini_nc=False)).to(device)
denoise_model.eval()
whisper_model = whisper.load_model("large").to(device)
whisper_model.eval()

def split_audio(tmpdirname, video, chunk_size=120):
    """
    Split audio into chunks of chunk_size
    """
    path = opj(tmpdirname, 'noisy_chunks')
    os.makedirs(path)
    audio = AudioFileClip(video.name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as audio_fp:
        audio.write_audiofile(audio_fp.name)

        # round duration to the next whole integer
        for i, chunk in enumerate(np.arange(0, audio.duration, chunk_size)):
            ffmpeg_extract_subclip(audio_fp.name, chunk, min(chunk + chunk_size, audio.duration),
                                targetname=opj(path, f'{i:09}.wav'))
    return audio.duration


def get_speakers(tmpdirname, use_auth_token=True):
    files = find_audio_files(opj(tmpdirname, 'noisy_chunks'))
    dset = Audioset(files, with_path=True,
                    sample_rate=denoise_model.sample_rate, channels=denoise_model.chin, convert=True)
    
    loader = distrib.loader(dset, batch_size=1)
    distrib.barrier()

    print('removing noise...')
    enhanced_chunks = []
    with tempfile.TemporaryDirectory() as denoised_tmpdirname:
        for data in loader:
            noisy_signals, filenames = data
            noisy_signals = noisy_signals.to(device)
            
            with torch.no_grad():
                wav = denoise_model(noisy_signals).squeeze(0)
            wav = wav / max(wav.abs().max().item(), 1)

            name = opj(denoised_tmpdirname, filenames[0].split('/')[-1])
            torchaudio.save(name, wav.cpu(), denoise_model.sample_rate)
            enhanced_chunks.append(name)

        print('reassembling chunks...')
        clips = [AudioFileClip(c) for c in sorted(enhanced_chunks)]
        final_clip = concatenate_audioclips(clips)
        cleaned_path = opj(tmpdirname, 'cleaned.wav')
        final_clip.write_audiofile(cleaned_path)
        print('identifying speakers...')

        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=use_auth_token)
    
        return str(pipeline({'uri': '', 'audio': cleaned_path})).split('\n'), cleaned_path

def get_subtitles(timecodes, clened_audio_path, language='en'):
    if(device == 'cpu'):
        options = whisper.DecodingOptions(language=language, fp16=False)
    else:
        options = whisper.DecodingOptions(language=language)

    timeline = {}
    prev_speaker = None
    prev_start = 0
    for line in timecodes:
        start, end = re.findall(r'\d{2}:\d{2}:\d{2}.\d{3}', line)
        start = str_to_seconds(start)
        end = str_to_seconds(end)
        speaker = re.findall(r'\w+$', line)[0]

        # extract a segment of the audio for a speaker
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as audio_fp:
            ffmpeg_extract_subclip(clened_audio_path, start, end,
                                    targetname=audio_fp.name)

            # load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio(audio_fp.name)
            audio = whisper.pad_or_trim(audio)  
            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
            # decode the audio
            result = whisper.decode(whisper_model, mel, options)

            if(speaker == prev_speaker):
                timeline[prev_start]['text'] += f' <{seconds_to_str(start)}>{result.text}'
                timeline[prev_start]['end'] = end
            else:
                timeline[start] = { 'end': end, 
                                    'speaker': speaker,
                                    'text': f'<v.{speaker}>{speaker}</v>: {result.text}'}
                prev_start = start

            prev_speaker = speaker

    return timeline

def str_to_seconds(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def seconds_to_str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    # with milliseconds
    return f'{int(h):02}:{int(m):02}:{s:06.3f}'
    

def timeline_to_vtt(timeline):
    vtt = 'WEBVTT\n\n'
    for start in sorted(timeline.keys()):
        end = timeline[start]['end']
        text = timeline[start]['text']
        vtt += f'{seconds_to_str(start)} --> {seconds_to_str(end)}\n'
        vtt += text+'\n\n'
    return vtt

def calc_speaker_percentage(timeline, duration):
    percentages = []
    end = 0
    for start in sorted(timeline.keys()):
        if(start > end):
            percentages.append(['Background', 100*(start-end)/duration])
        end = timeline[start]['end']
        speaker = timeline[start]['speaker']
        percentages.append([speaker, 100*(end-start)/duration])
    return percentages
