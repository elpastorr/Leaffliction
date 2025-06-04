import os
import sys
# from plantcv import plantcv as pcv

def tranfo(image_path: str):



if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise Exception("Input should be: Transformation.py path/to/image")
    except Exception as e:
        print(e.__class__.__name__, e)
        exit(0)
    try:
        if not os.path.exists(sys.argv[1]):
            raise FileNotFoundError(sys.argv[1])
    except Exception as e:
        print(e.__class__.__name__, e)
        exit(0)

    tranfo(sys.argv[1])
