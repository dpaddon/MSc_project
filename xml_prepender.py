#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:47:53 2018

@author: daniel
"""

import os

xml_path = '/Users/daniel/Documents/UCL/Project/Data/annotation-data/annotations_august/VC2010_worms10_food1-10_Set1_Pos4_Ch6_20122017_150107/'


files = [f for f in os.listdir(xml_path) if not f.startswith('.')]

for file in files:
    with open(xml_path + str(file), 'r') as original: data = original.read()
    with open(xml_path + str(file), 'w') as modified: modified.write('<?xml version="1.0" encoding="utf-8" standalone="yes"?> <Data>' + data + '</Data>')
