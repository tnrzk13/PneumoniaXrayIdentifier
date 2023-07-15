from fastai.vision.all import *
import gradio as gr

learn = load_learner('chestXray.pkl')

categories = ('No Pneumonia', 'Pneumonia')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
path = 'images/'
examples = [f'{path}Pneumonia 1.jpeg', f'{path}Pneumonia 2.jpeg', f'{path}Pneumonia 3.jpeg', f'{path}No Pneumonia 1.jpeg', f'{path}No Pneumonia 2.jpeg', f'{path}No Pneumonia 3.jpeg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)