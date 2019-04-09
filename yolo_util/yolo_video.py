import sys
import argparse
from yolo_util.yolo import YOLO, detect_video
from PIL import Image
import os
import matplotlib.pyplot as plt

def detect_img(yolo, img_path, img_name,output_path=""):
    #while True:
        #img = input('Input image filename:')
        #print(img_path)
        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
            #continue
        else:
            r_image,box = yolo.detect_image(image)
            #r_image.show()
            output_name = os.path.splitext(os.path.basename(img_name))[0]
            plt.imsave(os.path.join('yolo_util/crops2/', "{}_yolo.png".format(output_name)), r_image)
            plt.imshow(r_image)
            plt.show()
            return box
    

def detect_multiple(yolo, img_path, output_path=""):
    from glob import glob
    import os
    import os.path
    # test_data_path = './data/crops2'
    results = {}
    imgs = glob(os.path.join(img_path, "*.png"))
    for i in range(len(imgs)):
    	img_name = imgs[i]
    	box = detect_img(yolo,imgs[i],img_name,output_path)
    	# print(box)
    	results[i] = box
    return results




FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        ##'--image', default=False, action="store_true",
        "--image", nargs='?', type=str,required=False,default=False,action='store',
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    parser.add_argument(
        "--path", nargs='?', type=str,required=False,default="",action='store',
        help='Image detection mode, will ignore all positional arguments'
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        #if "input" in FLAGS:
         #   print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)), FLAGS.image, FLAGS.output)
    elif "path" in FLAGS:
        detect_multiple(YOLO(**vars(FLAGS)), FLAGS.path, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
