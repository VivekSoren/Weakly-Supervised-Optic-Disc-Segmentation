import os 
import shutil

if __name__ == '__main__':
    OUT_SUBSET = f'disc'
    OUT_DIR = os.path.join(f'input', OUT_SUBSET)
    os.makedirs(os.path.join(OUT_DIR, f'images'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, f'masks'), exist_ok=True)

    INPUT_DIR = f'disc_pipeline_outputs/crops/cup'

    img_list, mask_list = [], []
    for img in os.listdir(INPUT_DIR):
        if img.endswith(".jpg"):
            img_list.append(img)
        if img.endswith(".png"):
            mask_list.append(img)
    print(img_list[:10], mask_list[:10])
    print(len(img_list), len(mask_list))

    for img, mask in zip(img_list, mask_list):
        try: 
            shutil.copy(os.path.join(INPUT_DIR, img), os.path.join(OUT_DIR, f'images'))
            shutil.copy(os.path.join(INPUT_DIR, mask), os.path.join(OUT_DIR, f'masks'))
        except Exception as e: 
            print(f'An error occured')
        # break
