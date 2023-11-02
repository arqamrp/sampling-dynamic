import gradio as gr
import pandas as pd
from gradio import Interface, components
import matplotlib.cm as cm
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
from math import factorial
from tqdm import tqdm
from sklearn.cluster import DBSCAN

def create_trajectory_gif(k1=3, k2=20, p=0.1, u1=1, u2=2, u3=2.3, timesteps=6, grid_interval=0.1):
    k1 = int(round(k1))
    k2 = int(round(k2))
    timesteps = int(round(timesteps))

    # Computes the probability of a strategy being best response to a given sample of k
    def xtp1(x1, x2, k):
        probs = np.zeros(3)
        for i in range(k):
            for j in range(k-i):
                if 1.* i/k *u1 > 1.*j/k *u2 and 1.* i/k*u1 > 1.*(k-i-j)/k *u3:
                    probs[0] +=  factorial(k)/(factorial(j)* factorial(i) * factorial(k-i-j) ) *(x1)**i * (x2)**j * (1-x1-x2)**(k-i-j)
                elif 1.* j/k *u2 > 1.*i/k *u1 and 1.* j/k*u2 > 1.*(k-i-j)/k *u3:
                    probs[1] +=  factorial(k)/(factorial(j)* factorial(i) * factorial(k-i-j) ) *(x1)**i * (x2)**j * (1-x1-x2)**(k-i-j)
                else:
                    probs[2] +=  factorial(k)/(factorial(j)* factorial(i) * factorial(k-i-j) ) *(x1)**i * (x2)**j * (1-x1-x2)**(k-i-j)
        return probs

    def W(x1, x2):
        return p*xtp1(x1, x2, k1) + (1-p)* xtp1(x1, x2, k2)

    # Define the system of differential equations
    def system(Y, t=0):
        x1, x2 = Y
        w = W(x1, x2)
        dx1dt = w[0] - x1
        dx2dt = w[1] - x2
        return [dx1dt, dx2dt]
    
    # time space
    t = np.linspace(0, timesteps, int(timesteps*5))
    grid_points = [(x, y) for x in np.arange(0, 1.1, grid_interval) for y in np.arange(0, 1.1, grid_interval) if x + y <= 1]

    # Generating a color map
    colors = cm.rainbow(np.linspace(0, 1, len(grid_points)))

    final_points = []
    frames = []
    for t_val in tqdm(t):
        plt.figure(figsize=(12, 8))
        for ic, color in zip(grid_points, colors):
            t_segment = np.linspace(0, t_val, 100)  # 100 points for each trajectory segment
            sol = odeint(system, ic, t_segment)
            plt.plot(sol[:, 0], sol[:, 1], color=color, alpha=0.5)
            if t_val == t[-1]:  # If it's the last timestep, collect the final point
                final_points.append(sol[-1])
        
        plt.title(f"Attractor plot at time = {t_val:.2f}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid(True)
        plt.tight_layout()
        
        # Save the current frame to a file and add to the list of frames
        filename = f"./temp/temp_frame_{int(t_val*10):04d}.png"
        plt.savefig(filename)
        frames.append(Image.open(filename))
        plt.close()

    # Save frames as a GIF
    gif_filename = './trajectory_animation2.gif'
    frames[0].save(gif_filename, save_all=True, append_images=frames[1:], loop=0, duration=100)

    # Cleanup temporary frame files
    for frame in frames:
        frame.close()

    # Cluster the final points using DBSCAN
    clustering = DBSCAN(eps=0.1, min_samples=1).fit(final_points)
    cluster_labels = clustering.labels_

    # Determine the stable points by averaging the points in each cluster
    stable_points = []
    for label in set(cluster_labels):
        points_in_cluster = np.array(final_points)[cluster_labels == label]
        stable_point = np.around(np.mean(points_in_cluster, axis=0), decimals = 2)
        stable_points.append(stable_point)

    list_of_lists = [list(arr) + [1 - sum(arr)] for arr in stable_points]
    df = pd.DataFrame(list_of_lists, columns=['x1', 'x2', 'x3'])

    return gif_filename, df



iface = gr.Interface(
    fn= create_trajectory_gif,
    inputs=[
        gr.Number( label="Sample Size 1"),
        gr.Number( label="Sample Size 2"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Proportion of Cohort 1 (with sample size 1)"),
        gr.Number(label="Payoff u1"),
        gr.Number(label="Payoff u2"),
        gr.Number(label="Payoff u3"),
        gr.Number(label="Timesteps"),
        gr.Number(label="Grid resolution"),
    ],
    outputs=[
        gr.Image(label="Simulation GIF"),
        gr.Dataframe(type="pandas", label="Stable equilibria")
    ],
    examples=[ 
        [3, 20, 0.1, 1, 2, 2.3, 6, 0.1]
    ]
)

iface.launch()