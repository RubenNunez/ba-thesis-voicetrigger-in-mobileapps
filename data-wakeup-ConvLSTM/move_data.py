import os
import shutil

base_directory_osx = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM"
base_directory_win = "D:\\Projects\\ba-thesis-voicetrigger-in-mobileapps\\data-wakeup-ConvLSTM"

# Set the base directory
base_directory = base_directory_win

# Define known prefixes
known_prefixes = ["Hey_FOOBY", "Hello_FOOBY", "Hi_FOOBY", "OK_FOOBY", "FOOBY"]

for filename in os.listdir(base_directory):
    
    # Check if the file is a .ogg file
    if not filename.endswith('.ogg'):
        continue  # Skip this iteration and move to the next file

    # Determine the appropriate prefix
    prefix = "other"  # Default to "other"
    for known_prefix in known_prefixes:
        if known_prefix in filename:
            prefix = known_prefix
            break
    
    # Construct the destination folder path based on the prefix and ensure it exists
    destination_folder = os.path.join(base_directory, prefix)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Construct the full source and destination paths
    source_path = os.path.join(base_directory, filename)
    destination_path = os.path.join(destination_folder, filename)
    
    # Move the file
    shutil.move(source_path, destination_path)

print("Files moved successfully!")
