import os
import dill

def dump(file_obj,file_name:str,path:str):
    if not os.path.isdir(path):
        print(f'Path {path} provided not exists; So not dumping the file')
        return
    print(f'creating {path}/{file_name}.pk')
    with open(f'{path}/{file_name}.pk','wb') as fp:
        dill.dump(file_obj,fp)
    print(f'created {path}/{file_name}.pk')

def load(file_path):
    if not os.path.exists(file_path):
        print('Given path doesn\'t exist')
        return None
    print(f'Reading {file_path}')
    with open(file_path,'rb') as fp:
        read_file = dill.load(fp)
    return read_file
