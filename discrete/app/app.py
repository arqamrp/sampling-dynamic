
from PIL import Image

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd

from gradio import Interface, components

import random 

class population:
    
    # k1: sampling size for cohort 1, p1: proportion of cohort 1
    # k2: sampling size for cohort 2, 1-p1: proportion of cohort 2
    # Payoff matrix:
    # [a 0]
    # [0 b]
    
    def __init__(self, num_agents, k1, k2, p1, a, b, x):
        self.size = num_agents
        self.agentlist = []
        for i in range(int(round(num_agents*p1))):
            self.agentlist.append(agent(k1, a, b, x))
        for i in range(int(round(num_agents*p1)), int(round(num_agents))):
            self.agentlist.append(agent(k2, a, b, x))

    def proportion(self):
        n = 0
        for i in self.agentlist:
            n+= i.curr_strategy
        return (n/self.size)
        
    def update(self):
        for i in self.agentlist:
            i.update(self)
        
        n = 0
        for i in self.agentlist:
            i.step()
            n+= i.curr_strategy

        print(n/self.size)

    # 1: strat 1
    # 0: strat 2

class agent():
    def __init__(self, sampling_size, a, b, x):
        self.k = sampling_size
        self.a = a
        self.b = b
        self.curr_strategy = 1 if random.random() < x else 0
        self.next_strategy = -1
    
    def best_response(self, sample_size, n1):
        strat1_payoff = n1/sample_size * self.a
        strat2_payoff = (sample_size-n1)/sample_size * self.b
        return (strat1_payoff > strat2_payoff)
    
    def update(self, population):
        n1 = 0
        sample = random.sample(population.agentlist, self.k)
        for i in sample:
            n1 += i.curr_strategy
        
        self.next_strategy = int(self.best_response(self.k, n1))
        
    def step(self):
        self.curr_strategy = self.next_strategy

    def simulate(n_steps, pop_size, k1, k2, p1, a, b, x):
        pop = population(pop_size, k1, k2, p1, a, b, x)
        pop.proportion()
        for i in range(n_steps):
            pop.update()



def simulate_and_plot(n_steps, pop_size, k1, k2, p1, a, b, x):
    images = []
    proportions = []
    
    n_steps = int(round(n_steps))
    k1 = int(round(k1))
    k2 = int(round(k2))
    pop_size = int(round(pop_size))

    pop = population(pop_size, k1, k2, p1, a, b, x)
    for i in range(int(round(n_steps))):
        pop.update()
        prop = pop.proportion()
        proportions.append([i, prop])
        
        plt.figure(figsize=(5, 1))
        plt.barh([0], [prop], color='blue')
        plt.barh([0], [1-prop], color='red', left=prop)
        plt.axis('off')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0)
        img.seek(0)
        img_b64 = base64.b64encode(img.read()).decode()
        images.append(img_b64)
        plt.close()
        
    df_proportions = pd.DataFrame(proportions, columns=['Time Step', 'Proportion'])
    # img = Image.open(io.BytesIO(base64.b64decode(images[-1]))) 
    # return img, df_proportions
    pil_images = [Image.open(io.BytesIO(base64.b64decode(img_b64))) for img_b64 in images]

    # Create a GIF
    gif_path = "./static/gifs/anim.gif" 
    pil_images[0].save(gif_path,
                   save_all=True, append_images=pil_images[1:], 
                   duration= 5000./n_steps , loop=0)


    # Return the GIF and DataFrame
    return f"./static/gifs/anim.gif", df_proportions

             

iface = gr.Interface(
    fn=simulate_and_plot,
    inputs=[
        gr.Number(label="Number of Steps"),
        gr.Number( label="Population Size"),
        gr.Number( label="Sample Size 1"),
        gr.Number( label="Sample Size 2"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Proportion of Cohort 1 (with sample size 1)"),
        gr.Number(label="Payoff a"),
        gr.Number(label="Payoff b"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Initial Proportion of Strategy 1")
    ],
    outputs=[
        gr.Image(label="Simulation GIF"),
        gr.Dataframe(type="pandas", label="Proportions at Each Time Step")
    ],
    examples=[ 
        [10, 1e4, 3, 500, 0.4, 2.4, 1, 0.1 ]
    ]
)

iface.launch()