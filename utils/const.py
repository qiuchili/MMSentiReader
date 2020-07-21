# -*- coding: utf-8 -*-

FEATURE_NAME_DIC = {
        'cmumosi': {'acoustic': ['COVAREP','CMU_MOSI_COVAREP'],
                      'linguistic': ['glove_vectors','CMU_MOSI_TimestampedWordVectors'],
                      'visual': ['FACET 4.1','CMU_MOSI_VisualFacet_4.1'],
                      'sentiment':['Opinion Segment Labels','CMU_MOSI_Opinion_Labels'],
                      'textual':['words','CMU_MOSI_TimestampedWords']
                      },
        'cmumosei': {'acoustic': ['FACET 4.2','CMU_MOSEI_VisualFacet42'],
#                      'linguistic': ['glove_vectors','CMU_MOSEI_TimestampedGloveVectors'],
                      'visual': ['COVAREP','CMU_MOSEI_COVAREP'],
                      'sentiment': ['Sentiment Labels','CMU_MOSEI_LabelsSentiment'],
                      'emotion': ['Emotion Labels','CMU_MOSEI_LabelsEmotions'],
                      'textual': ['words','CMU_MOSEI_TimestampedWords']
                     },
        'pom': {'acoustic':['COVAREP', 'POM_COVAREP'],
                      'linguistic': ['glove_vectors','POM_TimestampedWordVectors'],
                      'visual': ['FACET 4.2','POM_Facet_42'],
#                      'visual': ['FACET 4.1','POM_Facet_41'],
                      'sentiment': ['video level sentiment', 'POM_Labels_Video_Level_Sentiment'],
                      'persuasion': ['video level persuasion', 'POM_Labels_Video_Level_Persuasion'],
                      'textual': ['words','POM_TimestampedWords'],
                      'personality': ['video level personality traits','POM_Labels_Video_Level_Personality_Traits'],
                      'phonetic': ['phonemes','POM_TimestampedPhones']
                     },
        'iemocap': {'acoustic' : ['OpenSMILE','IEMOCAP_openSMILE_IS09'],
                    'textual' : ['words','IEMOCAP_RIGHT_TimestampedWords'],
                    'visual' : ['OpenFace','IEMOCAP_OpenFace_Rightside'],
                    'emotion': ['Emotion Labels','IEMOCAP_RIGHT_EmotionLabels'],
                    'visual_2' : ['OpenFace 2','IEMOCAP_OpenFace_Leftside'],
                    'textual_2' : ['words 2','IEMOCAP_LEFT_TimestampedWords'],
                    'emotion_2': ['Emotion Labels 2','IEMOCAP_LEFT_EmotionLabels'],
                    'speaker_markers':['*','%']
                    }
        }
    