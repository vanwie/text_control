#!/usr/bin/env python3

import pyaudio
import Queue as queue
import threading
import time
import numpy as np
from concurrent import futures
import soundfile as sf
import tempfile
import shutil
import os
import requests
from uuid import uuid4
import math
import datetime
import subprocess
import fuzzy
from Levenshtein import distance

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

# Add some exports that the commands file wants
os.environ['KEYPRESS'] = 'xvkbd -xsendevent -secure -text'

mic_buffer = queue.Queue()
stt_buffer = queue.Queue()
text_out = queue.Queue()

done = threading.Event()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 2205 # 100ms

CAL_SIZE=50

class AudioSegmenter(threading.Thread): 
    _damping = 0.15
    _threshold = 500
    _ratio = 1.5
    _timeout = 0.75 
    _max_duration = 90

    
    def __init__(self, auto_calibrate=False):
        super(AudioSegmenter, self).__init__()
        self._auto_calibrate = auto_calibrate

    def run(self): 
        state = 0 # default to not recording
        first =True 
        data = np.array([])

        hist_0 = np.ones(CAL_SIZE)*self._threshold/self._ratio
        hist_1 = np.ones(CAL_SIZE)*self._threshold*self._ratio
        h0_idx = 0
        h1_idx = 0

        prev_chunk = np.array([])
        while not done.is_set():
            try:
                chunk = mic_buffer.get(timeout=0.1)
            except queue.Empty:
                continue
            power_avg = math.sqrt(sum(map(lambda x: 1.0*x*x, chunk))/len(chunk))

            if first:
                first=False
                continue

            if state == 0 and power_avg >= self._threshold:
                print(datetime.datetime.now(), "Voice recording started")
                N_lt = 0
                state = 1
                data = np.append(prev_chunk, chunk)
            elif state == 1:

                data = np.append(data, chunk)
                if power_avg < self._threshold:
                    N_lt += 1
                else:
                    if self._auto_calibrate: 
                        hist_1[h1_idx] = power_avg
                        h1_idx=(h1_idx+1)%CAL_SIZE
                    N_lt = 0

                if (N_lt*CHUNK/RATE) > self._timeout or len(data) > self._max_duration*RATE:
                    print(datetime.datetime.now(), "Voice recording completed. Enqueued %0.1fs worth of audio for processing" % (len(data)/RATE))
                    stt_buffer.put(data)
                    state = 0
                    data = np.array([])
            else:
                if self._auto_calibrate: 
                    hist_0[h0_idx] = power_avg
                    h0_idx=(h0_idx+1)%CAL_SIZE
                    self._threshold = (np.median(hist_1)+np.median(hist_0))/2
                    print("hist_0: %r" % hist_0)
                    print("hist_1: %r" % hist_1)
                    print("m0: %r m1: %r threshold: %r" % (np.median(hist_0), np.median(hist_1), self._threshold))
            prev_chunk=chunk

class GoogleSpeechToText(threading.Thread):

    def run(self): 
        client = speech.SpeechClient()
        config = types.RecognitionConfig(
                    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=RATE,
                    language_code='en-US')
        while not done.is_set():
            try:
                data = stt_buffer.get(timeout=0.1)
            except queue.Empty:
                continue

            print(datetime.datetime.now(), "Requesting transcription of %d data frames" % data.shape[0])
            audio = types.RecognitionAudio(content=data.tobytes())
            response = client.recognize(config, audio)
            print(datetime.datetime.now(), "Transcription complete: %r" % response) 
            for result in response.results:
                text = result.alternatives[0].transcript
                print(datetime.datetime.now(), "Text: %r" % text) 
                if text: 
                    text_out.put(text)
                else:
                    print(datetime.datetime.now(), 'Decoding complete. No text returned')


#class AWSSpeechToText(threading.Thread):
#    def run(self):
#        s3 = boto3.client('s3')
#        try: 
#            s3.create_bucket(Bucket=S3_BUCKET, CreateBucketConfiguration={ 'LocationConstraint': 'us-west-2'})
#        except Exception as e: 
#            if not 'BucketAlreadyOwnedByYou' in str(e): 
#                print("E: %r" % str(e))
#                raise
#            
#        tempdir = tempfile.mkdtemp()
#        try: 
#            while not done.is_set(): 
#                try:
#                    data = stt_buffer.get(timeout=0.1)
#                except queue.Empty:
#                    continue
#
#                #fn = os.path.join(tempdir, '%d.flac' % 1000*time.time())
#                job = str(uuid4())
#                fn = '%s.flac' % job
#                print(datetime.datetime.now(), "Uploading file: %s" % fn)
#                pn = os.path.join(tempdir, fn)
#                sf.write(pn, data, RATE, 'PCM_16')
#                s3.upload_file(pn, S3_BUCKET, fn)
#
#                transcribe = boto3.client('transcribe')
#                transcribe.start_transcription_job(
#                        TranscriptionJobName=job,
#                        LanguageCode='en-US',
#                        MediaFormat='flac',
#                        Media={ 'MediaFileUri': 'https://s3-{region}.amazonaws.com/{bucket}/{file}'.format(region='us-west-2', bucket=S3_BUCKET, file=fn) }
#                        )
#
#                while not done.is_set():
#                    status = transcribe.get_transcription_job(TranscriptionJobName=job)
#                    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
#                        break
#                    time.sleep(0.1)
#                if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED': 
#                    uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
#                    try:
#                        transcript = requests.get(uri).json()['results']['transcripts'][0]['transcript']
#                        text_out.put(text)
#                    except:
#                        print(datetime.datetime.now(), "Unable to get transcript from job %r" % job)
#
#        finally: 
#            shutil.rmtree(tempdir)
#
class TextProcessor(threading.Thread): 
    commands = { 'exit' : None, 'cancel' : None }
    nysiis = None
    nysiis_distance_threshold = 100

    def __init__(self, command_file):
        super(TextProcessor, self).__init__()
        self.parse_command_file(command_file)

    def parse_command_file(self, filename):
        loaded=0
        with open(filename, 'r') as f:
            for i, raw_line in enumerate(f):
                try: 
                    line = raw_line[:raw_line.index('#')]
                except ValueError: 
                    line = raw_line
                line=line.strip()
                if line:
                    if ':' in line:
                        phrase, cmd = line.split(':', 1)
                        phrase, cmd = phrase.strip().lower(), cmd.strip()
                        if phrase and cmd:
                            self.commands[phrase] = cmd
                            loaded+=1
                        else:
                            print(datetime.datetime.now(), "Invalid command file line %d: %r" % (i, raw_line))
                    else: 
                        print(datetime.datetime.now(), "Invalid command file line %d: %r" % (i, raw_line))

            print(datetime.datetime.now(), "Loaded %d commands from file. %d total commands known." % (loaded, len(self.commands)))
        self.nysiis = { fuzzy.nysiis(k) : k for k in self.commands.keys() }

    def match_phrase(self, spoken): 
        spoken = spoken.lower()
        # Attempt 1:  direct match
        if spoken in self.commands:
            print(datetime.datetime.now(), "Matched phrase %r with confidence of 100.0" % spoken)
            return spoken

        # Attempt 2: phonetic
        spoken = fuzzy.nysiis(spoken)
        distances = { distance(spoken, k)/float(len(spoken)) : v for k, v in self.nysiis.items() }
        min_dist = min(distances.keys())
        conf = 95-45*min_dist/self.nysiis_distance_threshold
        if conf >= self.nysiis_distance_threshold:
            print(datetime.datetime.now(), "Matched phrase %r with confidence of %0.1f" % (distances[min_dist], conf))
            return distances[min_dist]
        else:
            print(datetime.datetime.now(), "Matched phrase %r with confidence of %0.1f; confidence too low. Ignoring!" % (distances[min_dist], conf))

        return None

    def run(self):
        proc = None
        while not done.is_set(): 
            try:
                spoken = text_out.get(timeout=0.1)
            except queue.Empty:
                continue

            if done.is_set():
                return


            print(datetime.datetime.now(), "Matching spoken phrase %r..." % (spoken))
            phrase = self.match_phrase(spoken)
            if phrase: 
                if self.commands[phrase] is None: 
                    print(datetime.datetime.now(), "Matched phrase %r, will perform appropriate action." % phrase)
                    if phrase == 'exit':
                        done.set()
                    elif phrase == 'cancel': 
                        if proc.poll() is None:
                            proc.kill()
                else:
                    print(datetime.datetime.now(), "Matched phrase %r, will execute command %r" % (phrase,  self.commands[phrase]))
                    if proc and proc.poll() is None:
                        print(datetime.datetime.now(), "The previous command is still running!")
                    else:
                        proc = subprocess.Popen(self.commands[phrase], shell=True)
            

def enqueue_audio(in_data, frame_count, time_info, status):
    audio_data = np.fromstring(in_data, dtype=np.int16)
    mic_buffer.put(audio_data)
    return (in_data, pyaudio.paContinue)

if __name__ == '__main__': 
    done.clear()
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=enqueue_audio)
    seg = AudioSegmenter()
    stt = GoogleSpeechToText()
    tp = TextProcessor(os.path.join(os.path.dirname(__file__), 'commands'))

    stream.start_stream()
    seg.start()
    stt.start()
    tp.start()

    try: 
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")
        done.set()
    stream.stop_stream()
    seg.join()
    stt.join()
    tp.join()



