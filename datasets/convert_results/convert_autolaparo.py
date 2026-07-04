import os
import numpy as np
import torch
cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

import numpy as np
import os
import glob

phases = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nproc_per_node', type=int, default=4, help='Number of GPUs used during inference')
parser.add_argument('--main_path', type=str, default="./results/", help='Path to result directory')
args = parser.parse_args()

nproc_per_node = args.nproc_per_node
main_path = args.main_path

# Dynamically load result files based on nproc_per_node
file_paths = [os.path.join(main_path, f"{i}.txt") for i in range(nproc_per_node)]
all_lines = []
for fp in file_paths:
    with open(fp) as f:
        all_lines.append(f.readlines())
     
anns_path = main_path + "/phase_annotations"
pred_path = main_path + "/prediction"

create_folder_if_not_exists(anns_path)
create_folder_if_not_exists(pred_path)


# Generate ground-truth annotation files
for i in range(15, 22):
    with open(
        anns_path + "/video-{}.txt".format(str(i)), "w"
    ) as f:
        f.write("Frame")
        f.write("\t")
        f.write("Phase")
        f.write("\n")
        # Verify all files have the same number of lines
        for k in range(1, nproc_per_node):
            assert len(all_lines[0]) == len(all_lines[k])
        for j in range(1, len(all_lines[0])):
            for k in range(nproc_per_node):
                temp = all_lines[k][j].split()
                if temp[1] == "{}".format(str(i)):
                    f.write(str(temp[2]))  # phase_annotations
                    f.write("\t")  # phase_annotations
                    f.write(str(temp[-1]))  # phase_annotations
                    f.write("\n")  # phase_annotations
            
            
# Reload files to generate prediction results
all_lines = []
for fp in file_paths:
    with open(fp) as f:
        all_lines.append(f.readlines())

# Generate prediction result files
for i in range(15, 22):
    print(i)
    with open(
        pred_path + "/video-{}.txt".format(str(i)), "w"
    ) as f:  # phase_annotations
        f.write("Frame")
        f.write("\t")
        f.write("Phase")
        f.write("\n")
        # Verify all files have the same number of lines
        for k in range(1, nproc_per_node):
            assert len(all_lines[0]) == len(all_lines[k])
        for j in range(1, len(all_lines[0])):
            for k in range(nproc_per_node):
                line_strip = all_lines[k][j].strip()  # prediction
                data = np.fromstring(
                    line_strip.split("[")[1].split("]")[0], dtype=np.float32, sep=","
                )  # prediction
                data = data.argmax()  # prediction
                temp = all_lines[k][j].split()
                if temp[1] == "{}".format(str(i)):
                    f.write(str(temp[2]))  # prediction
                    f.write('\t')  # prediction
                    f.write(str(data))  # prediction
                    f.write('\n')  # prediction
            