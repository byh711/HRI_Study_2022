# -*- coding:utf-8 -*-
'''
All kind of settings (MS Azure API keys, setting the log file, taking *.txt files)
Setting global variables, read data files (*_utterances.txt, *_keywords.txt, settings.txt)
'''
################### load packages
# generaL behavior
import os, json
import pandas as pd
from collections import defaultdict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# for logging:
import warnings, logging

#for TTS
import azure.cognitiveservices.speech as speechsdk

################ set API keys
# MS Auzre / used at 'tts.py' and 'asr.py' files
speech_key, service_region = "b41c17f0032346748d5fc7f72f7c9db2", "koreacentral"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
audio_config = speechsdk.audio.AudioConfig(device_name="{0.0.1.00000000}.{fddaf6be-d0d0-49ef-9225-fe76132285d4}")
################ set API keys endssss
################### load packages end

################ set logging
# Ignoring the warnings
warnings.filterwarnings(action='ignore')

# Making the 'logs' folder if the folder does not exist
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
    
if not os.path.exists("./Script/"):
    os.mkdir("./Script/")
    
if not os.path.exists("./Planning/"):
    os.mkdir("./Planning/")  

# Set up log file written format ex) 01:39:09
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
    level=logging.DEBUG,
    filename='./logs/dm.log',
    # encoding='utf-8',
    datefmt="%H:%M:%S",
    #stream=sys.stderr,
)

logging.getLogger("chardet.charsetprober").disabled = True
################ set logging end

################ set global variables and constants

################# PLACEHOLDERS (All used at 'parse_tree.py' file)
### Set up repetition delay count
REP_DELAY_AMT = 3 # prevent too much talk
LOCAL_REP_DELAY_AMT = REP_DELAY_AMT
### repetition delay count ends

### Other placeholders
HUNGRY = 50
STARVING = 30
INJURED = 50
DYING = 30
SANITY_SAFE = 80
SANITY_DANGER = 30
CHATTINESS = 3.0 # talkative or not
### Other placeholders end
################## PLACEHOLDERS END

################## Define dictionaries, lists (used at 'state_changed.py file or 'parse_tree.py' file)
### For repetition delay
# Set up state_list that are used at the 'state_changed.py' file
state_list = ['Phase', 'Hunger_AVATAR', 'Health_AVATAR', 'Sanity_AVATAR', 'Curr_Active_Item_AVATAR',
              'Curr_Equip_Hands_AVATAR', 'Attack_Target_AVATAR',  'Defense_Target_AVATAR', 'Food_AVATAR',
              'Tool_AVATAR', 'Lights_AVATAR', 'Is_Light_AVATAR', 'Is_Monster_AVATAR']

# Concatenate above 'state_list' and 'REP_DELAY_AMT' as a dictionary
rep_delay_states = dict.fromkeys(state_list, REP_DELAY_AMT)
### repetition delay ends


### planning start
data_list = list()
prediction_result_data = list()

columns = [
    'OS_timestamp', 'Game_Runtime', 'Phase', 'Unnamed: 3', 'Hunger_AVATAR',
    'Health_AVATAR', 'Sanity_AVATAR', 'AVATAR_ID', 'AVATAR_Xloc', 'AVATAR_Yloc',
    'AVATAR_Zloc', 'Curr_Inv_Cnt_AVATAR', 'Curr_Active_Item_AVATAR', 'Curr_Equip_Hands_AVATAR', 'Attack_Target_AVATAR',
    'Defense_Target_AVATAR', 'Recent_attacked_AVATAR', 'Food_AVATAR', 'hunger:health:sanity', 'Tool_AVATAR',
    'twigs:flint', 'Lights_AVATAR', 'log:rock:grass:twigs', 'Is_Light_AVATAR', 'Is_Monster_AVATAR',
    'Monster_num_AVATAR', 'Unnamed: 26', 'Hunger_PLAYER', 'Health_PLAYER', 'Sanity_PLAYER',
    'Player ID', 'Player_Xloc', 'Player_Yloc', 'Player_Zloc', 'Curr_Inv_Cnt_PLAYER',
    'Curr_Active_Item_PLAYER', 'Curr_Equip_Hands_PLAYER', 'Attack_Target_PLAYER', 'Defense_Target_PLAYER', 'Recent_attacked_PLAYER',
    'Food_PLAYER', 'hunger:health:sanity.1', 'Tool_PLAYER', 'twigs:flint.1', 'Lights_PLAYER',
    'log:rock:grass:twigs.1', 'Is_Light_PLAYER', 'Is_Monster_PLAYER', 'Monster_num_PLAYER', 'Unnamed: 49',
    'Distance', 'Unnamed: 51',
]

planning_columns = ['Game_Runtime','Phase', 'Hunger_AVATAR', 'Health_AVATAR', 'Sanity_AVATAR', 
               'AVATAR_Xloc', 'AVATAR_Zloc', 'Curr_Inv_Cnt_AVATAR',  'Curr_Active_Item_AVATAR', 
               'Curr_Equip_Hands_AVATAR','Attack_Target_AVATAR', 'Defense_Target_AVATAR',
               'Recent_attacked_AVATAR', 'Food_AVATAR', 'Is_Light_AVATAR', 'Monster_num_AVATAR', 
               'twig_A', 'flint_A', 'log_A', 'rock_A', 'grass_A','Hunger_PLAYER', 'Health_PLAYER', 
               'Sanity_PLAYER', 'Player_Xloc','Player_Zloc', 'Curr_Inv_Cnt_PLAYER', 
               'Curr_Active_Item_PLAYER', 'Curr_Equip_Hands_PLAYER', 'Attack_Target_PLAYER', 
               'Defense_Target_PLAYER', 'Recent_attacked_PLAYER', 'Food_PLAYER',
               'Is_Light_PLAYER', 'Monster_num_PLAYER', 'twig_P', 'flint_P',
               'log_P', 'rock_P', 'grass_P', 'Distance']


path1 = './Working_code/saved_models_ext/DontStarve_PlannedSpeech_Model_Class1'
model1 = load_model(path1)

path2 = './Working_code/saved_models_ext/DontStarve_PlannedSpeech_Model_Class2'
model2 = load_model(path2)

path3 = './Working_code/saved_models_ext/DontStarve_PlannedSpeech_Model_Class3'
model3 = load_model(path3)

path4 = './Working_code/saved_models_ext/DontStarve_PlannedSpeech_Model_Class4'
model4 = load_model(path4)

path5 = './Working_code/saved_models_ext/DontStarve_PlannedSpeech_Model_Class5'
model5 = load_model(path5)

path6 = './Working_code/saved_models_ext/DontStarve_PlannedSpeech_Model_Class6'
model6 = load_model(path6)

path7 = './Working_code/saved_models_ext/DontStarve_PlannedSpeech_Model_Class7'
model7 = load_model(path7)


# model = model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics.AUC()])
game_time = 0
pred_time = 0
csvcounter = 1
### planning end

data = dict()
initial_state = dict()
status = defaultdict(list)
################## Define dictionaries, lists end

scriptcounter = 1

game_start_time = 0
utt_start_time = 0
utt_finish_time = 0

list_utter = list()

threshold = 0.999

planning_path=''
script_path=''

synthesize_utt_check = 0

# read settings from 'settings.txt' file
with open('settings.txt', 'r', encoding='utf-8') as settingsfile:
    exec(settingsfile.read())

# read list of utterances from the 'data/*_utterances.txt' file
with open('data/' + THIS_LANGUAGE + '_utterances.txt', 'r', encoding='utf-8') as utterancefile:
    utt_data = utterancefile.read()
    RESPONSE_UTTS = json.loads(utt_data, strict=False)

# read list of keywords from the 'data/*_keywords.txt' file
with open('data/' + THIS_LANGUAGE + '_keywords.txt', 'r', encoding='utf-8') as keywordfile:
    asrkeys_data = keywordfile.read()
    ASR_KEYS_UTTS = json.loads(asrkeys_data, strict=False)
    
# read list of planning utterances from the 'data/*_planning_utterances.txt' file
with open('data/' + THIS_LANGUAGE + '_planning_utterances.txt', 'r', encoding='utf-8') as planningfile:
    asrkeys_data = planningfile.read()
    PLANNING_UTTS = json.loads(asrkeys_data, strict=False)

# read list of utterances priority median value from the 'data/*_priority_utterances.xlsx' file as dictionary type
# Read the Excel file under data folder. Currently, use 'category' and 'median' columns for Self-generated utterances
priority_utterance = pd.read_excel('data/' + THIS_LANGUAGE + '_priority_utterances.xlsx', usecols = ['category', 'median'])
# Making 'category' and 'median' dataframe type to list type
category_list = priority_utterance['category'].values.tolist()
median_list = priority_utterance['median'].values.tolist()
# Concatenate both to dictionary
priority_utterance_score = dict(zip(category_list,median_list))

# read list of keyword priority median value from the 'data/*_priority_keywords.xlsx' file as dictionary type
# Read the Excel file under data folder. Currently, use the 'median' column for keywords
priority_keyword = pd.read_excel('data/' + THIS_LANGUAGE + '_priority_keywords.xlsx', usecols = ['median'])
# Making 'median' dataframe type to list type
value_list = priority_keyword.values.tolist()
# Making 'median' list-list type to list (ex. [[3.0]] -> [3.0])
value_list = sum(value_list, [])
# Concatenate 'keywords' and 'median' to dictionary
priority_keyword_score = dict(zip(ASR_KEYS_UTTS.keys(),value_list))

INTERFACE_FILEFOLDER = INTERFACE_FOLDER + INTERFACE_FILE
################ set global variables and constants end
