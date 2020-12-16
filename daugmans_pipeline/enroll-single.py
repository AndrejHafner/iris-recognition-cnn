##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import argparse, os
from time import time
from scipy.io import savemat

from fnc.extractFeature import extractFeature


if __name__ == '__main__':


    #------------------------------------------------------------------------------
    #	Argument parsing
    #------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str,
                        help="Path to the file that you want to verify.")

    parser.add_argument("--temp_dir", type=str, default="./templates/temp/",
                        help="Path to the directory containing templates.")

    args = parser.parse_args()


    ##-----------------------------------------------------------------------------
    ##  Execution
    ##-----------------------------------------------------------------------------
    start = time()
    args.file = "../CASIA1/1/001_1_1.jpg"

    # Extract feature
    print('>>> Enroll for the file ', args.file)
    template, mask, file = extractFeature(args.file)

    # Save extracted feature
    basename = os.path.basename(file)
    out_file = os.path.join(args.temp_dir, "%s.mat" % (basename))
    savemat(out_file, mdict={'template':template, 'mask':mask})
    print('>>> Template is saved in %s' % (out_file))

    end = time()
    print('>>> Enrollment time: {} [s]\n'.format(end-start))