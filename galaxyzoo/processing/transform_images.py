import galaxyzoo.processing.api as di
import os
import json
import matplotlib.pyplot as plt

ROOT = "/Users/jaidevd/GitHub/kaggle/galaxyzoo"
RAW_IMG_DIR = os.path.join(ROOT,'images_training_rev1')
NEW_IMG_DIR = os.path.join(ROOT, 'processed_images')

defective = []

def get_processed_img_path(path):
    bn = os.path.basename(path)
    image_id = bn.split('.')[0]
    return os.path.join(NEW_IMG_DIR, image_id+'.jpg')

def process_image(x):
    org_region = di.get_largest_region(x)
    rr1, cc1   = di.get_bbox_center(org_region.bbox)
    rotated_x  = di.rotate_largest_region(x)
    _x         = cc1 - x.shape[1]/2
    _y         = rr1 - x.shape[0]/2
    cc2, rr2   = di.point_rotate(_x, _y, -org_region.orientation)
    rr2       += rotated_x.shape[0]/2
    cc2       += rotated_x.shape[1]/2
    cropped    = di.crop_around_centroid(rotated_x, rr2, cc2)
    return cropped

def main():
    all_files = os.listdir(RAW_IMG_DIR)
    i = 0
    for image in all_files:
        try:
            filename = os.path.basename(image)
            path = os.path.join(RAW_IMG_DIR, filename)
            x = plt.imread(path)[:,:,0]
            cropped = process_image(x)
            new_path = get_processed_img_path(path)
            plt.imsave(new_path, cropped, cmap=plt.cm.gray)
        except Exception, err:
            defective.append((image, str(err)))
            print "Error:", err
        i += 1
        if i % 1000 == 0:
            print i
    with open(os.path.join(ROOT,'defective_files.json'),'w') as f:
        json.dump(defective, f)

if __name__ == "__main__":
    main()
