"""
Run this code after processing a chunck of files. It will check if they have been processed correctly (i.e. the mock sources have been stored in the specified folder) and, if so, move them to the specified folder.
"""

import os
import shutil
import argparse

try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3

print(os.getcwd())

def read_inputs():
    parser = argparse.ArgumentParser(description='Move already processed files to another folder.')
    
    parser.add_argument('input_folder', type=str, help='Name of the folder where the files to process are.')
    parser.add_argument('processed_folder', type=str, help='Name of the folder to store the already processed files.')
    parser.add_argument('output_folder', type=str, help='Folder where the outputs are stored.')
    parser.add_argument('constants_file', type=str, help='Name of the file with the constants (.ini file, like those used by AGAMA).')
    args = parser.parse_args()
    
    return args

def read_ini(filename):
    ini = RawConfigParser()
    ini.optionxform=str  # do not convert key to lowercase
    ini.read(filename)
    return ini


### RUN CODE ###

if __name__ == "__main__":

    #read input from the terminal
    args = read_inputs()
    params = read_ini(args.constants_file)

    input_folder = args.input_folder
    output_folder = args.output_folder
    processed_folder = args.processed_folder

    print(input_folder,processed_folder,output_folder,params["Basic"]["output_filename"])

    processed_list = [f[f.find(params["Basic"]["glon_colname"]):f.find(params["Basic"]["glat_colname"])+len(params["Basic"]["glat_colname"])+6] \
                      for f in os.listdir(output_folder) if f.startswith(params["Basic"]["output_filename"])]

    print(len(processed_list),len(os.listdir(output_folder)))

    count = 0
    for infile in os.listdir(input_folder):
        aux = infile[infile.find(params["Basic"]["glon_colname"]):infile.find(params["Basic"]["glat_colname"])+len(params["Basic"]["glat_colname"])+6]

        if aux in processed_list:
            count += 1
            shutil.move(input_folder+infile,processed_folder+infile)

    print(count)