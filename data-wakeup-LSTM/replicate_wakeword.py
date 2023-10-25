import os
import argparse
import shutil

def main(args):
    # List files in wakeword directory
    files = os.listdir(args.wakewords_dir)
    
    for file in files:
        if file.endswith(".wav"):
            for i in range(args.copy_number):
                # Define source and destination file paths
                src_file = os.path.join(args.wakewords_dir, file)
                dest_file = os.path.join(args.copy_destination, str(i) + "_" + file)

                # Copy and rename the file directly
                shutil.copy(src_file, dest_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to replicate the wakeword clips by a bunch of times.
    """
    )
    parser.add_argument('--wakewords_dir', type=str, default=None, required=True,
                        help='directory of clips with wakewords')
    parser.add_argument('--copy_destination', type=str, default=None, required=True,
                        help='directory of the destinations of the wakewords copies')
    parser.add_argument('--copy_number', type=int, default=100, required=False,
                        help='the number of copies you want')

    args = parser.parse_args()

    main(args)
