from flask import Flask, render_template, request
from flask_cors import cross_origin
from PIL import Image
import base64
import io
import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import numpy as np
from PIL import Image
import json

from model import *

application = Flask(__name__)
app = application


IMG_DIR = "/Users/wuw/downloads/archive"

G_net = Generator()
D_net = Discriminator()
device = torch.device('cpu')

input_transform = tt.Compose([
    tt.Resize((64,64)),
    tt.ToTensor(),
    tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

@app.before_first_request
def init():
    print("here")
    G_net.load_state_dict(torch.load("models/g_model", map_location=device))
    D_net.load_state_dict(torch.load("models/d_model", map_location=device))



def random_pic_from_db():
    dataset = ImageFolder(IMG_DIR, transform=tt.Compose([
        tt.Resize((64, 64))
    ]))
    
    rand_idx = np.random.randint(low=0, high=len(dataset), size=64)
    dst = Image.new('RGB', (64*8, 64*8))
    for i, idx in enumerate(rand_idx):
        r = i // 8
        c = i % 8
        img, _ = dataset[idx]
        img = np.array(img)
        img = Image.fromarray(np.array(img), 'RGB')
        dst.paste(img, (r*64, c*64))
    return dst

def generate_random_art(genre, batch):
    latent_noise = torch.randn(batch, latent_size, 1, 1)
    label = torch.full((batch,), genre)
    label_onehot = F.one_hot(label, 10).unsqueeze(-1).unsqueeze(-1)
    g_input = torch.cat((latent_noise, label_onehot), dim=1)
    fake_image = G_net(g_input)
    grid = make_grid(denorm(fake_image.detach()), nrow=int(np.sqrt(batch))).permute(1,2,0).numpy()
    return grid

def recognize_image_from_user(img):
    img = input_transform(img).unsqueeze(0)
    print(img.shape)
    D_net.eval()
    with torch.no_grad():
        pred, label = D_net(img)
    pred_class = F.softmax(label, dim=1)
    return pred_class

@app.route("/dataset")
@cross_origin()
def dataset():
    im = random_pic_from_db()
    #im.save('../frontend/public/images/random_image.jpg')
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return {"imgValue": encoded_img_data.decode('utf-8')}


@app.route("/model", methods=["POST"])
@cross_origin()
def generate_plot():
    props = request.json
    genre = int(props['genre'])
    num = int(props['count'])
    im = generate_random_art(genre, num)

    data = io.BytesIO()
    im = Image.fromarray((im * 255).astype(np.uint8))
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return {"imgValue": encoded_img_data.decode('utf-8')}

@app.route("/upload_img", methods=["GET", "POST"])
@cross_origin()
def upload_img():
    if request.method == 'POST':
        file = request.files['file']
        im = Image.open(file)
        pred_class = recognize_image_from_user(im).squeeze(0)
        winner = torch.argmax(pred_class)
        return {"pred": pred_class.tolist(),"winner": winner.item()}
    else:
        img = Image.open("sample_pic.jpg")
        data = io.BytesIO()
        img.save(data, 'JPEG')
        encoded_img_data = base64.b64encode(data.getvalue())
        return {"imgValue": encoded_img_data}

@app.route("/members", methods=['POST'])
def members():
    return {"members": ['mem1', 'mem2']}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)