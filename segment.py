import os
os.environ["MPLBACKEND"] = "Agg" # Use non-GUI backend for headless environments
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from torchvision import transforms
from mmseg.structures import SegDataSample
from segearthov3_segmentor import SegEarthOV3Segmentation

# change the path to the image that you want to segment
img_path = '/kaggle/input/datasets/dummyirl/vogelsbergkreis-lautertal-dop20/DOP20_32_525_5604_1_he.jpg'

# change the name_list to the classes that you want to segment
# each class can have similar names separated by commas, e.g. 'tree, forest'
# each class should be in different line in the name_list.txt file
name_list = [
    'background',
    'road',
    'building',
    'solar panel',
    'grass',
    'tree',
    'farmland',
    'clutter'
]

# define a color map for the segmentation classes
COLOR_MAP = np.array([
    [0,0,0],        # background
    [255,255,255],  # road
    [0,0,255],      # building
    [180,0,255],    # solar panel
    [0,180,0],      # grass
    [0,255,0],      # tree
    [255,255,0],    # farmland
    [255,0,0],      # clutter
], dtype=np.uint8)

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list) - 1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')

img = Image.open(img_path).convert("RGB")
img_tensor = transforms.Compose([
    transforms.ToTensor(),
])(img).unsqueeze(0).to('cuda') # This variable is only a placeholder; the actual data is read within the model. (To be optimized)

data_sample = SegDataSample()
img_meta = {
    'img_path': img_path,
    'ori_shape': img.size[::-1]
}
data_sample.set_metainfo(img_meta)

# change the model parameters according to your needs
model = SegEarthOV3Segmentation(
    type='SegEarthOV3Segmentation',
    model_type='SAM3',
    classname_path='./configs/my_name.txt',
    prob_thd=0.1,
    bg_idx=0,
    confidence_threshold=0.1,
    slide_stride=768,
    slide_crop=1024,
)

seg_pred = model.predict(img_tensor,data_samples=[data_sample])
seg_pred = seg_pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)

seg_rgb = COLOR_MAP[np.clip(seg_pred,0,len(COLOR_MAP) - 1)]
overlay_alpha = 0.5

os.makedirs("output", exist_ok=True)

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(10, 7), dpi=300)
fig.subplots_adjust(wspace=0)
ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title(Path(img_path).name,fontsize=10,fontweight='bold')
ax[1].imshow(img)
ax[1].imshow(seg_rgb,alpha=overlay_alpha)
ax[1].axis('off')
ax[1].set_title(f'Segmentation Result (α={overlay_alpha})',fontsize=10,fontweight='bold'
)
fig.tight_layout(rect=[0, 0.15, 1, 1])

legend_elements = []
for class_name, color in zip(name_list,COLOR_MAP):
    legend_elements.append(Patch(facecolor=color / 255.0,edgecolor='black',label=class_name))

fig.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.1),
    frameon=False,
    ncol=min(4, len(name_list)),
    prop={'size': 9,'weight': 'bold'}
)

meta_text = (
    f"img_size = {img.width}x{img.height}    "
    f"prob_thd = {model.prob_thd}    "
    f"conf_thd = {model.confidence_threshold}\n"
    f"slide_stride = {model.slide_stride}    "
    f"slide_crop = {model.slide_crop}"
)

fig.text(
    0.5,
    0.025,
    meta_text,
    ha='center',
    va='bottom',
    fontsize=10,
    family='monospace'
)

# Save the figure
output_path = os.path.join("output",f"{Path(img_path).stem}_segmented.png")
plt.savefig(output_path)
plt.close()  # Close the figure to free memory
# plt.show() # commented for headless environments
print(f"Saved: {output_path}")