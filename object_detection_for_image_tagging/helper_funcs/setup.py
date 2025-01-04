import os
import subprocess

# Set up directory structure & make darknet
def setup(cwd, basewd):
        # Make folders if they don't already exist
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
                # Download pretrained YOLOv3 weights for Open Images
                weightsPath = 'https://pjreddie.com/media/files/yolov3-openimages.weights'
                subprocess.Popen(['wget', weightsPath])

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
            print("\nDarknet successfully installed! Move onto next steps to do object detection with YOLOv3.")

        return cwd

if __name__ == "__main__":
    import sys
    setup(sys.argv[1], sys.argv[2])