# ------------------------------------------------------------------------------------
# VK-OOD
# Interactive Outlier Detection Enables Robust Deep Multimodal Analysis
# ------------------------------------------------------------------------------------
# Modified from METER (https://github.com/zdou0830/METER)
# Copyright (c) 2021 Microsoft Corporation. All Rights Reserved.
# Licensed under MIT License(https://github.com/zdou0830/METER/blob/main/LICENSE)
# ------------------------------------------------------------------------------------
# Modified from ViLT (https://github.com/dandelin/ViLT)
# Copyright 2021-present NAVER Corp. All Rights Reserved.
# Licensed under Apache 2.0(https://github.com/dandelin/ViLT/blob/master/LICENSE)
# ------------------------------------------------------------------------------------
# Modified from CLIP(https://github.com/openai/CLIP)
# Copyright (c) 2021 OpenAI. All Rights Reserved.
# Licensed under MIT License(https://github.com/openai/CLIP/blob/main/LICENSE)
# ------------------------------------------------------------------------------------
# Modified from Swin-Transformer(https://github.com/microsoft/Swin-Transformer)
# Copyright (c) Microsoft Corporation. All Rights Reserved.
# Licensed under MIT License(https://github.com/microsoft/Swin-Transformer/blob/main/LICENSE)
# ------------------------------------------------------------------------------------


import sng_parser
from pprint import pprint
import spacy
import json
import requests
import numpy as np
import random
import statistics as st
import math
from sklearn import metrics
import sklearn
import pandas as pd
import torch

def parse(sent):
    graph = sng_parser.parse(sent)
    return graph

def get_conceptnet(word):
    relations = ["AtLocation","PartOf","IsA","UsedFor","HasA"]
    word = word.lower()
    original_word= word
    if (len(word.split())>0):
        word = word.replace(" ","_")
    
    concept_list = {}
    for rel in relations:
      words = []
      url = "http://api.conceptnet.io/query?node=/c/en/%s&rel=/r/%s&start=/c/en/%s&limit=5"%(word,rel,word)
      #print(url)
      obj = requests.get(url).json()

      for edge in obj['edges']:
          word2 = edge['end']['label'] 
          words.append(word2)
      concept_list[rel] = words
                   
    return concept_list


def mahalanobis(x,mu,phi=1): 
    return(-0.5*(1/phi)*torch.inner(x-mu,x-mu))


def rescaled_GEM_score(x,mean,phi=1):
    energy=0

    for mu in mean:
        energy+=torch.exp(mahalanobis(x,mu,phi))

    return energy