import torch
from os.path import join as opj
from PIL import Image
from model import Gnet
from torchvision import transforms
import numpy as np


model_path = './checkpoints/best_checkpoint.pth'
file_name = 'test.png'
file_path = f'./test/{file_name}'
output_path = f'./output'


model = Gnet.SGRU().cuda()
ckp = torch.load(model_path)
model.load_state_dict(ckp['model'])
model = model.eval()

img = Image.open(file_path).convert('L').resize((256, 256))
img = transforms.ToTensor()(img).unsqueeze(0)

with torch.no_grad():
    img = img.cuda()
    output = model(img).squeeze()  # [9, 3, H, W]

output = output.detach().cpu().numpy()
output = np.minimum(np.maximum(output, 0.0), 255.0)
output = output.astype(np.uint8).transpose(0, 2, 3, 1)
for i in range(output.shape[0]):
##    img_fake_rgb = transforms.ToPILImage()(output[i, ...])
    img_fake_rgb = Image.fromarray(output[i, ...])
    # img_fake_rgb.show()
    img_fake_rgb.save(opj(output_path, f'result_{file_name.split(".")[0]}_{i}.png'))

