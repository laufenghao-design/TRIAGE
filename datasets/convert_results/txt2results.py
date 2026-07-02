import numpy as np
import os
import glob

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)

Dataset2id = {
    "Cholec80": {"VIDEO_START_ID": 41, "VIDEO_END_ID": 80, "VIDEO_NAME_FORMAT": "video{id}"},
    "AutoLaparo": {"VIDEO_START_ID": 15, "VIDEO_END_ID": 21, "VIDEO_NAME_FORMAT": "video{id}"},
    "CATARACTS": {"VIDEO_START_ID": 1, "VIDEO_END_ID": 20, "VIDEO_NAME_FORMAT": "video{id}"},
    "PmLR50": {"VIDEO_START_ID": 1, "VIDEO_END_ID": 5, "VIDEO_NAME_FORMAT": "video{id}"},
    "M2CAI16-Workflow": {"VIDEO_START_ID": 28, "VIDEO_END_ID": 41, "VIDEO_NAME_FORMAT": "video_{id}"}
}
# Configuration
Dataset = "M2CAI16-Workflow"  # Dataset name
main_path = './results/'  # Path to result directory

# Mode: True = auto-mapping, False = fixed-number mode
AUTO_MAPPING_MODE = False

# Parameters for fixed-number mode (used only when AUTO_MAPPING_MODE=False)
VIDEO_START_ID = Dataset2id[Dataset]["VIDEO_START_ID"]
VIDEO_END_ID = Dataset2id[Dataset]["VIDEO_END_ID"]
VIDEO_NAME_FORMAT = Dataset2id[Dataset]["VIDEO_NAME_FORMAT"]

# Auto-find all numerically-named txt files
txt_files = sorted(glob.glob(os.path.join(main_path, "[0-9]*.txt")))
print(f"Found {len(txt_files)} txt files: {[os.path.basename(f) for f in txt_files]}")

anns_path = main_path + "/phase_annotations"
pred_path = main_path + "/prediction"

create_folder_if_not_exists(anns_path)
create_folder_if_not_exists(pred_path)

# Read all txt files
all_lines = []
for txt_file in txt_files:
    with open(txt_file) as f:
        lines = f.readlines()
        all_lines.append(lines)

# Set video processing range based on mode
if AUTO_MAPPING_MODE:
    # Auto-mapping mode: extract video IDs from all files and create mapping
    video_id = set()
    for lines in all_lines:
        for i in lines[1:]:
            parts = i.split()
            if len(parts) > 1:
                video_name = parts[1]
                # Extract video name (remove 'video' prefix if present)
                if video_name.startswith('video_'):
                    video_name = video_name[6:]  # Remove "video_" prefix
                elif video_name.startswith('video'):
                    video_name = video_name[5:]  # Remove "video" prefix
                video_id.add(video_name)
    
    video_id = sorted(video_id)
    id_to_video = dict(zip(range(1, len(video_id) + 1), video_id))
    print("Auto-mapping mode - Video ID mapping:", id_to_video)
    video_range = range(1, len(video_id) + 1)
else:
    # Fixed-number mode: use specified video ID range
    print(f"Fixed-number mode - Processing video IDs: {VIDEO_START_ID} to {VIDEO_END_ID}")
    video_range = range(VIDEO_START_ID, VIDEO_END_ID + 1)
    id_to_video = None

# Generate phase_annotations files
for i in video_range:
    if AUTO_MAPPING_MODE:
        video_name = id_to_video[i]
        output_filename = f"video-{i}.txt"
    else:
        video_name = str(i)
        output_filename = f"video-{i}.txt"
    
    # Use dict to collect current video data, auto-deduplicate
    video_data_dict = {}
    
    # Iterate all txt files
    for lines in all_lines:
        for j in range(1, len(lines)):
            temp = lines[j].split()
            file_video_name = temp[1]
            # Handle different video name formats
            expected_video_name = VIDEO_NAME_FORMAT.format(id=video_name)
            if file_video_name == expected_video_name or file_video_name == video_name:
                frame_num = int(temp[2])
                # M2CAI16 uses index 11 for phase, others use last element
                if Dataset == "M2CAI16-Workflow":
                    phase = int(temp[11])
                else:
                    phase = int(temp[-1])
                # Use frame number as key; later data overwrites previous duplicates
                video_data_dict[frame_num] = phase
    
    # Convert to list and sort by frame number
    video_data = [(frame_num, phase) for frame_num, phase in video_data_dict.items()]
    video_data.sort(key=lambda x: x[0])
    
    # Write to file
    with open(os.path.join(anns_path, output_filename), "w") as f:
        f.write("Frame\tPhase\n")
        for frame_num, phase in video_data:
            f.write(f"{frame_num}\t{phase}\n")

# Generate prediction files
for i in video_range:
    if AUTO_MAPPING_MODE:
        video_name = id_to_video[i]
        output_filename = f"video-{i}.txt"
        print(f"Processing video-{i} (mapped to: {video_name})")
    else:
        video_name = str(i)
        output_filename = f"video-{i}.txt"
        print(f"Processing video-{i}")

    # Use dict to collect current video prediction data, auto-deduplicate
    video_pred_data_dict = {}
    
    # Iterate all txt files
    for lines in all_lines:
        for j in range(1, len(lines)):
            temp_line = lines[j].strip()
            temp = lines[j].split()
            
            file_video_name = temp[1]
            # Handle different video name formats
            expected_video_name = VIDEO_NAME_FORMAT.format(id=video_name)
            if file_video_name == expected_video_name or file_video_name == video_name:
                try:
                    # Extract prediction data
                    data = np.fromstring(
                        temp_line.split("[")[1].split("]")[0], 
                        dtype=np.float32, 
                        sep=",",
                    )
                    predicted_phase = data.argmax()
                    frame_num = int(temp[2])
                    # Use frame number as key; later data overwrites previous duplicates
                    video_pred_data_dict[frame_num] = predicted_phase
                except (IndexError, ValueError) as e:
                    print(f"Error processing line {j}: {e}")
                    continue
    
    # Convert to list and sort by frame number
    video_pred_data = [(frame_num, predicted_phase) for frame_num, predicted_phase in video_pred_data_dict.items()]
    video_pred_data.sort(key=lambda x: x[0])
    
    # Write to file
    with open(os.path.join(pred_path, output_filename), "w") as f:
        f.write("Frame\tPhase\n")
        for frame_num, predicted_phase in video_pred_data:
            f.write(f"{frame_num}\t{predicted_phase}\n")

print("Processing complete!")
