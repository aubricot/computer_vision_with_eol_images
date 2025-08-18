import os
import subprocess
import shutil

# Set up directory structure & make darknet
def setup_darknet(basewd = "/content/drive/MyDrive/train", folder = "darknet"):
        # Make folders if they don't already exist
        cwd = basewd + '/' + folder
        if not os.path.exists(cwd):
                os.makedirs(basewd)
                os.chdir(basewd)
                print("\nBuilding directory structure\n")
                # Clone darknet
                repoPath = 'https://github.com/AlexeyAB/darknet'
                subprocess.check_call(['git', 'clone', repoPath])
                # Compile darknet
                os.chdir(cwd)
                # Make folders for detection datafiles
                os.makedirs('data/imgs')
                os.makedirs('data/img_info')
                os.makedirs('data/results')
                
        # Change makefile to have GPU and OPENCV enabled
        os.chdir(cwd)
        print("\nEnabling GPU and OpenCV in makefile...")
        subprocess.call(['sed', '-i', 's/OPENCV=0/OPENCV=1/', 'Makefile'])
        subprocess.call(['sed', '-i', 's/GPU=0/GPU=1/', 'Makefile'])
        subprocess.call(['sed', '-i', 's/CUDNN=0/CUDNN=1/', 'Makefile'])
        subprocess.call(['sed', '-i', 's/CUDNN_HALF=0/CUDNN_HALF=1/', 'Makefile'])

        # Verify CUDA version (for using GPU)
        subprocess.call(['/usr/local/cuda/bin/nvcc', '--version'])

        # Make darknet
        print("\n~~~Making darknet...~~~\n")
        subprocess.call('make', stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        if os.path.exists('./darknet'):
            # Move weights file to darknet
            weights_path = cwd + '/' + 'yolov3-openimages.weights'
            shutil.move("/content/yolov3-openimages.weights", weights_path)
            print("\n\033[92m Darknet successfully installed! Move onto next steps to do object detection with YOLOv3.")

        return cwd, basewd