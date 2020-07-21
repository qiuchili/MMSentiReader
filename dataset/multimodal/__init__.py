# -*- coding: utf-8 -*-

from .cmusdk_reader import CMUMOSEIDataReader, CMUMOSIDataReader, POMDataReader, IEMOCAPDataReader

def setup(opt):
    if opt.dataset_name.lower() == 'cmumosei':
        reader = CMUMOSEIDataReader(opt)
    elif opt.dataset_name.lower() == 'cmumosi':
        reader = CMUMOSIDataReader(opt)
    elif opt.dataset_name.lower() == 'pom':    
        reader = POMDataReader(opt)
    elif opt.dataset_name.lower() == 'iemocap':    
        reader = IEMOCAPDataReader(opt)
    return reader