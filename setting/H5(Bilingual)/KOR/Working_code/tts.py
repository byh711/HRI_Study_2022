# -*- coding:utf-8 -*-
'''
Synthesize utterances using Microsoft Azure TTS SDK mainly from 'parse_tree.py' file and 'asr.py' file.
'''
import azure.cognitiveservices.speech as speechsdk

from Working_code import config as cf
import time
import csv
import datetime

# set API keys
speech_key, service_region = cf.speech_key, cf.service_region
speech_config = cf.speech_config

# set logger
logger = cf.logging.getLogger("__tts__")

# MS Azure Text to Speech(TTS) SDK synthesize the sentence
def synthesize_utt(utterance):
    # get a text and synthesize it
    logger.info('New utterance is:')
    logger.info(utterance)

    if utterance != None:
        cf.utt_start_time = round(time.time() - cf.game_start_time,5)
        
        # print result which utterance is selected
        print()
        print('**********' * 5)
        print('Agent: ', utterance)
        print('**********' * 5)
        print()
        cf.utt_finish_time = round(time.time() - cf.game_start_time,5)
        cf.utt_time_diff = cf.utt_finish_time - cf.utt_start_time
        
        cf.list_utter.append("Avatar")
        cf.list_utter.append(utterance)
        cf.list_utter.append(str(datetime.timedelta(seconds=cf.utt_start_time)).split(".")[0])
        
        if cf.scriptcounter == 1:
            with open(cf.filepath, encoding="utf-8-sig", newline='', mode="w") as f:
                writer = csv.writer(f)
                writer.writerow(['Speaker','Script','Time'])
                writer.writerow(cf.list_utter)
                
            cf.scriptcounter = cf.scriptcounter + 1
            
        elif cf.scriptcounter == 2:
            with open(cf.filepath, encoding="utf-8-sig", newline='', mode="a") as f:
                writer = csv.writer(f)
                writer.writerow(cf.list_utter)
            
        
        cf.list_utter = list()
        
        
        # MS Azure TTS / Synthesize text and make it to speak
        # Sample codes could be checked right below link
        # https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/quickstart/python/text-to-speech/quickstart.py
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = speech_synthesizer.speak_text_async(utterance).get()

        
        # If error occurs, print it and pass to work next utterance with no problem
        # Below is an error that is occurring for MS Azure
        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            pass
        '''
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        # use a player like pygame with mp3_fp
        # check out: https://github.com/pndurette/gTTS/issues/26
        '''
        # ToDo: put synthesizer and player in separate threads and queue