# -*- coding:utf-8 -*-
'''
Synthesize utterances using Microsoft Azure TTS SDK mainly from 'parse_tree.py' file and 'asr.py' file.
'''
import azure.cognitiveservices.speech as speechsdk
from Working_code import config as cf
import time
import datetime

from Working_code import asr

# set API keys
speech_key, service_region = cf.speech_key, cf.service_region
speech_config = cf.speech_config

# set logger
logger = cf.logging.getLogger("__tts__")


# MS Azure Text to Speech(TTS) SDK synthesize the sentence
def synthesize_utt(utterance):
    # get a text and synthesize it
    
    if cf.synthesize_utt_check == 0 and utterance != None:
        cf.utt_start_time = round(time.time() - cf.game_start_time,5)
        cf.synthesize_utt_check = 1
        logger.info(f"New utterance is: {utterance}")
        
        # MS Azure TTS / Synthesize text and make it to speak
        # Sample codes could be checked right below link
        # https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/quickstart/python/text-to-speech/quickstart.py
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        
        
        # print result which utterance is selected
        print()
        print('**********' * 5)
        print('Avatar: ', utterance)
        print('**********' * 5)
        print()
        
        cf.list_utter.append("Avatar")
        cf.list_utter.append(utterance)
        cf.list_utter.append(str(datetime.timedelta(seconds=cf.utt_start_time)).split(".")[0])
        
        cf.utt_finish_time = round(time.time() - cf.game_start_time,5)
        cf.list_utter.append(str(datetime.timedelta(seconds=cf.utt_finish_time)).split(".")[0])
        
        asr.asr_tts_excel()
        
        cf.list_utter = list()
        
        result = speech_synthesizer.speak_text_async(utterance).get()

        # If error occurs, print it and pass to work next utterance with no problem
        # Below is an error that is occurring for MS Azure
        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.info("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.info("Error details: {}".format(cancellation_details.error_details))
        '''
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        # use a player like pygame with mp3_fp
        # check out: https://github.com/pndurette/gTTS/issues/26
        '''
        # ToDo: put synthesizer and player in separate threads and queue
        cf.synthesize_utt_check = 0