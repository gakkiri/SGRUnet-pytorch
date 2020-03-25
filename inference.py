import torch
import os.path.join as opj
from PIL import Image
from model import Gnet
from dataset import transforms


model_path = './checkpoints/best_checkpoint.pth'
file_name = 'test.png'
file_path = f'./test{file_name}'
output_path = f'./output'


model = Gnet.SGRU()
model.load_state_dict(model_path)
model = model.cuda().eval()

img = Image.open(file_path).convert('L')
img = transforms.get_tsfm()(img).unsqueeze(0)

with torch.no_grad():
    img = img.cuda()
    output = model(img).squeeze()  # [9, 3, H, W]


for i in range(output.shape[0]):
    img_fake_rgb = transforms.ToPILImage()(output[i, ...])
    # img_fake_rgb.show()
    img_fake_rgb.save(opj(output_path, f'result_{file_name.split(".")[0]}_{i}'))

