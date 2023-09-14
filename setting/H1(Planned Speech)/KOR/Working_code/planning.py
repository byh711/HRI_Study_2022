
# -*- coding:utf-8 -*-
import time
import warnings
import csv
import pandas as pd
import numpy as np
import os
import datetime
import tensorflow as tf
import random
from concurrent.futures import ThreadPoolExecutor
import ast

from Working_code import config as cf
from Working_code import tts


# 현재 시간
starting_time = time.time()

# 경고 메세지 무시
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.autograph.experimental.do_not_convert(func=None)


logger = cf.logging.getLogger("__Planning__")

global X_arr

def pred(num):
    global X_arr
    if num == 1:
        prediction = cf.model1.predict(X_arr, verbose=0)
    elif num == 2:
        prediction = cf.model2.predict(X_arr, verbose=0)
    elif num == 3:
        prediction = cf.model3.predict(X_arr, verbose=0)
    elif num == 4:
        prediction = cf.model4.predict(X_arr, verbose=0)
    elif num == 5:
        prediction = cf.model5.predict(X_arr, verbose=0)
    elif num == 6:
        prediction = cf.model6.predict(X_arr, verbose=0)
    else:
        prediction = cf.model7.predict(X_arr, verbose=0)
    return prediction[0][1]



def get_planning_utterance_from_abstract(abstract):
    utt_list = ast.literal_eval(cf.PLANNING_UTTS[abstract]) 
    return random.choice(utt_list)   


def planning_excel():
    if cf.csvcounter == 1:
        with open(cf.planning_path, encoding="utf-8-sig", newline='', mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(['Night','Resources','Monster', 'Build Stuff', 'Social Interaction', 'Status','Environment', 'Category','Top acc','Pred time','Game time','Abv Thresh','Pred Cat','Text'])
            writer.writerow(cf.prediction_result_data)
        cf.csvcounter = cf.csvcounter + 1
        
    elif cf.csvcounter == 2:
        with open(cf.planning_path, encoding="utf-8-sig", newline='', mode="a") as f:
            writer = csv.writer(f)
            writer.writerow(cf.prediction_result_data)
    


def planning(row):
    # row는 test.csv 파일의 한 행
    global X_arr
    
    # append the row to the list
    cf.data_list.append(row)
    logger.info("Making raw data to list Done")
    
    # if the list items are 15 then execute below
    # test.csv에서 한 행씩 가져오는데 15행이 되면 실행시킨다.
    if len(cf.data_list) == 15:
        # start time recording
        start_time = time.time()

        # changing the list to dataframe format
        logger.info("Making 15 items to Dataframe")
        X_arr = pd.DataFrame(cf.data_list, columns=cf.columns)
        
        # cleaning the dataframe (changing string to numbers, int to float, etc )
        logger.info("Transform Dataframe to wanted shape")
        X_arr = data_cleaning(X_arr)

        X_arr = X_arr[cf.planning_columns]
        
        # make the 'RowID' column to the numpy array
        logger.info("Making data to array")
        
        X_arr = np.array(X_arr).reshape((1,15,41)).astype(float)

        # make prediction
        logger.info("Prediction Starts")

        
        # print(min(32, (os.cpu_count() or 1) + 4))
        sequence = [1,2,3,4,5,6,7]
        pool = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4))  
        cf.prediction_result_data = list(pool.map(pred, sequence))
        

        # get the max value and the index of that max value (if index is 3, then category number is 3)
        value = max(cf.prediction_result_data)
        max_index = cf.prediction_result_data.index(value)
        
        cf.prediction_result_data.append(max_index+1)
        cf.prediction_result_data.append(value)

        # Type -Night:1, Resources:2, Monster:3, Build Stuff:4, Social Interaction:5, Status:6, Environment:7
        # if the score is above 0.8 and index is as 0,1,2,3,4,5 then return synthesized tts
        if cf.prediction_result_data[7] == 1 and cf.prediction_result_data[8] > cf.threshold:
            logger.info("Prediction Result: Night")
            utterance = get_planning_utterance_from_abstract('Night')
            tts.synthesize_utt(utterance)
            
        elif cf.prediction_result_data[7] == 2 and cf.prediction_result_data[8] > cf.threshold:
            logger.info("Prediction Result: Resources")
            utterance = get_planning_utterance_from_abstract('Resources')
            tts.synthesize_utt(utterance)
            
        elif cf.prediction_result_data[7] == 3 and cf.prediction_result_data[8] > cf.threshold:
            logger.info("Prediction Result: Monster")
            utterance = get_planning_utterance_from_abstract('Monster')
            tts.synthesize_utt(utterance)
                        
        elif cf.prediction_result_data[7] == 4 and cf.prediction_result_data[8] > cf.threshold:
            logger.info("Prediction Result: Build Stuff")
            utterance = get_planning_utterance_from_abstract('Build Stuff')
            tts.synthesize_utt(utterance)
                        
        elif cf.prediction_result_data[7] == 5 and cf.prediction_result_data[8] > cf.threshold:
            logger.info("Prediction Result: Social Interaction")
            utterance = get_planning_utterance_from_abstract('Social Interaction')
            tts.synthesize_utt(utterance)

        elif cf.prediction_result_data[7] == 6 and cf.prediction_result_data[8] > cf.threshold:
            logger.info("Prediction Result: Status")
            utterance = get_planning_utterance_from_abstract('Status')
            tts.synthesize_utt(utterance)
                       
        elif cf.prediction_result_data[7] == 7 and cf.prediction_result_data[8] > cf.threshold:
            logger.info("Prediction Result: Environment")
            utterance = get_planning_utterance_from_abstract('Environment')
            tts.synthesize_utt(utterance)
                        
        # Append count and \n (for identifying previous result and current result at the csv) to the list
        logger.info("Prediction Result to CSV")

        # Add the time consumption to the list
        cf.pred_time = round((time.time() - start_time), 2)
        cf.game_time = str(datetime.timedelta(seconds=round(time.time() - cf.game_start_time,5))).split(".")[0]
        
        cf.prediction_result_data.append(cf.pred_time)
        cf.prediction_result_data.append(cf.game_time)
        
        if value > cf.threshold:
            cf.prediction_result_data.append(1)
            cf.prediction_result_data.append(max_index+1)
            cf.prediction_result_data.append(utterance)
        else:
            cf.prediction_result_data.append(0)
            cf.prediction_result_data.append('')
            cf.prediction_result_data.append('')
        
        planning_excel()

        cf.data_list.clear()
        cf.prediction_result_data.clear()
        pool.shutdown(wait=True)
        

def data_cleaning(dataframe_list):
    # Splitting combined columns
    # dataframe_list[['Affected_Hunger', 'Affected_Health', 'Affected_Sanity']] = dataframe_list['hunger:health:sanity'].str.split(':', n=2, expand=True)
    
    dataframe_list[['log_A', 'rock_A', 'grass_A', 'twig_A']] = dataframe_list['log:rock:grass:twigs'].str.split(':', n=3, expand=True)
    dataframe_list[['twigs_removed', 'flint_A']] = dataframe_list['twigs:flint'].str.split(':', n=1, expand=True)

    temp = []
    for i in range(len(dataframe_list['Phase'])):
        temp.append(0)

    if dataframe_list['log:rock:grass:twigs.1'].iloc[0] != '999':
        dataframe_list[['log_P', 'rock_P', 'grass_P', 'twig_P']] = dataframe_list['log:rock:grass:twigs.1'].str.split(':', n=3, expand=True)
    else:
        dataframe_list['log_P'] = temp
        dataframe_list['rock_P'] = temp
        dataframe_list['grass_P'] = temp
        dataframe_list['twig_P'] = temp
        
    
    if dataframe_list['twigs:flint.1'].iloc[0] != '999':
        dataframe_list[['twigs_removed.1', 'flint_P']] = dataframe_list['twigs:flint.1'].str.split(':', n=1, expand=True)
    else:
        dataframe_list['twigs_removed.1'] = temp
        dataframe_list['flint_P'] = temp

    # Change several columns: String to float/int
    # dataframe_list[['Affected_Hunger', 'Affected_Health', 'Affected_Sanity']] = dataframe_list[['Affected_Hunger', 'Affected_Health', 'Affected_Sanity']].astype(float)
    dataframe_list[['log_A', 'rock_A', 'grass_A', 'twig_A', 'flint_A']] = dataframe_list[['log_A', 'rock_A', 'grass_A', 'twig_A', 'flint_A']].astype(int)
    dataframe_list[['log_P', 'rock_P', 'grass_P', 'twig_P', 'flint_P']] = dataframe_list[['log_P', 'rock_P', 'grass_P', 'twig_P', 'flint_P']].astype(int)
    
    dataframe_list[['Player_Xloc', 'Player_Zloc']] = dataframe_list[['Player_Xloc', 'Player_Zloc']].astype(float)
    dataframe_list[['AVATAR_Xloc', 'AVATAR_Zloc']] = dataframe_list[['AVATAR_Xloc', 'AVATAR_Zloc']].astype(float)
    
    for i in range(len(dataframe_list)):
        
        # Phase -day:0, dusk:1, night:2
        if dataframe_list.loc[i, 'Phase'] == 'day':
            dataframe_list.loc[i, 'Phase'] = 0
        elif dataframe_list.loc[i, 'Phase'] == 'dusk':
            dataframe_list.loc[i, 'Phase'] = 1
        else:
            dataframe_list.loc[i, 'Phase'] = 2


        # Curr_Active_Item
        if dataframe_list.loc[i, 'Curr_Active_Item_AVATAR'] == 'nil':
            dataframe_list.loc[i, 'Curr_Active_Item_AVATAR'] = 0
        else:
            dataframe_list.loc[i, 'Curr_Active_Item_AVATAR'] = 1


        # Curr_Equip_Hands
        if dataframe_list.loc[i, 'Curr_Equip_Hands_AVATAR'] == 'nil':
            dataframe_list.loc[i, 'Curr_Equip_Hands_AVATAR'] = 0
        else:
            if dataframe_list.loc[i, 'Curr_Equip_Hands_AVATAR'].rsplit('-')[1].lstrip() == 'axe(LIMBO)':
                dataframe_list.loc[i, 'Curr_Equip_Hands_AVATAR'] = 1
            elif dataframe_list.loc[i, 'Curr_Equip_Hands_AVATAR'].rsplit('-')[1].lstrip() == 'pickaxe(LIMBO)':
                dataframe_list.loc[i, 'Curr_Equip_Hands_AVATAR'] = 2
            else:
                dataframe_list.loc[i, 'Curr_Equip_Hands_AVATAR'] = 3
                
                
        # Attack_Target
        if dataframe_list.loc[i, 'Attack_Target_AVATAR'] == 'nil':
            dataframe_list.loc[i, 'Attack_Target_AVATAR'] = 0
        else:
            dataframe_list.loc[i, 'Attack_Target_AVATAR'] = 1
            
            
        # Defense_Target
        if dataframe_list.loc[i, 'Defense_Target_AVATAR'] == 'nil':
            dataframe_list.loc[i, 'Defense_Target_AVATAR'] = 0
        else:
            dataframe_list.loc[i, 'Defense_Target_AVATAR'] = 1
            
            
        # Recent_attacked
        if dataframe_list.loc[i, 'Recent_attacked_AVATAR'] == 'nil':
            dataframe_list.loc[i, 'Recent_attacked_AVATAR'] = 0
        else:
            dataframe_list.loc[i, 'Recent_attacked_AVATAR'] = 1
            
        
        # Food -'No Food!':0, 'Less Food!':1, 'Fine!':2
        if dataframe_list.loc[i, 'Food_AVATAR'] == 'No Food!':
            dataframe_list.loc[i, 'Food_AVATAR'] = 0
        elif dataframe_list.loc[i, 'Food_AVATAR'] == 'Less Food!':
            dataframe_list.loc[i, 'Food_AVATAR'] = 1
        else:
            dataframe_list.loc[i, 'Food_AVATAR'] = 2
            

        # Is_Light -'No lights!':0, 'Lights nearby':1, 'Campfire nearby':2
        if dataframe_list.loc[i, 'Is_Light_AVATAR'] == 'No lights!':
            dataframe_list.loc[i, 'Is_Light_AVATAR'] = 0
        elif dataframe_list.loc[i, 'Is_Light_AVATAR'] == 'Lights nearby':
            dataframe_list.loc[i, 'Is_Light_AVATAR'] = 1
        else:
            dataframe_list.loc[i, 'Is_Light_AVATAR'] = 2
            
        
        
        # Curr_Active_Item
        if dataframe_list.loc[i, 'Curr_Active_Item_PLAYER'] == 'nil':
            dataframe_list.loc[i, 'Curr_Active_Item_PLAYER'] = 0
        elif dataframe_list.loc[i, 'Curr_Active_Item_PLAYER'] == '999':
            dataframe_list.loc[i, 'Curr_Active_Item_PLAYER'] = 0
        else:
            dataframe_list.loc[i, 'Curr_Active_Item_PLAYER'] = 1


        # Curr_Equip_Hands
        if dataframe_list.loc[i, 'Curr_Equip_Hands_PLAYER'] == 'nil':
            dataframe_list.loc[i, 'Curr_Equip_Hands_PLAYER'] = 0
        elif dataframe_list.loc[i, 'Curr_Equip_Hands_PLAYER'] == '999':
            dataframe_list.loc[i, 'Curr_Equip_Hands_PLAYER'] = 0
        else:
            if dataframe_list.loc[i, 'Curr_Equip_Hands_PLAYER'].rsplit('-')[1].lstrip() == 'axe(LIMBO)':
                dataframe_list.loc[i, 'Curr_Equip_Hands_PLAYER'] = 1
            elif dataframe_list.loc[i, 'Curr_Equip_Hands_PLAYER'].rsplit('-')[1].lstrip() == 'pickaxe(LIMBO)':
                dataframe_list.loc[i, 'Curr_Equip_Hands_PLAYER'] = 2
            else:
                dataframe_list.loc[i, 'Curr_Equip_Hands_PLAYER'] = 3
                
                
                
        # Attack_Target
        if dataframe_list.loc[i, 'Attack_Target_PLAYER'] == 'nil':
            dataframe_list.loc[i, 'Attack_Target_PLAYER'] = 0
        elif dataframe_list.loc[i, 'Attack_Target_PLAYER'] == '999':
            dataframe_list.loc[i, 'Attack_Target_PLAYER'] = 0
        else:
            dataframe_list.loc[i, 'Attack_Target_PLAYER'] = 1
            
            
        # Defense_Target
        if dataframe_list.loc[i, 'Defense_Target_PLAYER'] == 'nil':
            dataframe_list.loc[i, 'Defense_Target_PLAYER'] = 0
        elif dataframe_list.loc[i, 'Defense_Target_PLAYER'] == '999':
            dataframe_list.loc[i, 'Defense_Target_PLAYER'] = 0
        else:
            dataframe_list.loc[i, 'Defense_Target_PLAYER'] = 1
            
            
        # Recent_attacked
        if dataframe_list.loc[i, 'Recent_attacked_PLAYER'] == 'nil':
            dataframe_list.loc[i, 'Recent_attacked_PLAYER'] = 0
        elif dataframe_list.loc[i, 'Recent_attacked_PLAYER'] == '999':
            dataframe_list.loc[i, 'Recent_attacked_PLAYER'] = 0
        else:
            dataframe_list.loc[i, 'Recent_attacked_PLAYER'] = 1
            
        
        # Food -'No Food!':0, 'Less Food!':1, 'Fine!':2
        if dataframe_list.loc[i, 'Food_PLAYER'] == '999':
            dataframe_list.loc[i, 'Food_PLAYER'] = 0
        elif dataframe_list.loc[i, 'Food_PLAYER'] == 'No Food!':
            dataframe_list.loc[i, 'Food_PLAYER'] = 0
        elif dataframe_list.loc[i, 'Food_PLAYER'] == 'Less Food!':
            dataframe_list.loc[i, 'Food_PLAYER'] = 1
        else:
            dataframe_list.loc[i, 'Food_PLAYER'] = 2
            

        # Is_Light -'No lights!':0, 'Lights nearby':1, 'Campfire nearby':2
        if dataframe_list.loc[i, 'Is_Light_PLAYER'] == '999':
            dataframe_list.loc[i, 'Is_Light_PLAYER'] = 0
        elif dataframe_list.loc[i, 'Is_Light_PLAYER'] == 'No lights!':
            dataframe_list.loc[i, 'Is_Light_PLAYER'] = 0
        elif dataframe_list.loc[i, 'Is_Light_PLAYER'] == 'Lights nearby':
            dataframe_list.loc[i, 'Is_Light_PLAYER'] = 1
        else:
            dataframe_list.loc[i, 'Is_Light_PLAYER'] = 2
    
    dataframe_list = dataframe_list.replace('999',0)
    dataframe_list = dataframe_list.replace(999,0)
    return dataframe_list