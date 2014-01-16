import data_inspector as di
import os
import json
import matplotlib.pyplot as plt

RAW_IMG_DIR = os.path.join(os.getcwd(),'images_training')
NEW_IMG_DIR = os.path.join(os.getcwd(), 'processed_images')

defective = []

def get_processed_img_path(path):
    bn = os.path.basename(path)
    image_id = bn.split('.')[0]
    return os.path.join(NEW_IMG_DIR, image_id+'.png')

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
    for image in all_files:
        try:
            filename = os.path.basename(image)
            path = os.path.join(RAW_IMG_DIR, filename)
            x = plt.imread(path)[:,:,0]
            cropped = process_image(x)
            new_path = get_processed_img_path(path)
            plt.imsave(new_path, cropped, cmap=plt.cm.gray)
        except:
            defective.append(image)
        print image
    with open('defective_files.json','w') as f:
        json.dump(defective, f)

if __name__ == "__main__":
    main()
