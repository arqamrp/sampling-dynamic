from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
import random
import io
import base64
import time

# Your existing code
# ... (put your class definitions and simulate function here)
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
        for i in range(int(num_agents*p1)):
            self.agentlist.append(agent(k1, a, b, x))
        for i in range(int(num_agents*p1), num_agents):
            self.agentlist.append(agent(k2, a, b, x))

    def proportion(self):
        n = 0
        for i in self.agentlist:
            n+= i.curr_strategy
        print(n/self.size)
        
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

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate_route():
    # Get parameters from the form
    n_steps = int(request.form['n_steps'])
    pop_size = int(request.form['pop_size'])
    k1 = int(request.form['k1'])
    k2 = int(request.form['k2'])
    p1 = float(request.form['p1'])
    a = float(request.form['a'])
    b = float(request.form['b'])
    x = float(request.form['x'])
    
    # Simulate and capture pie chart images
    images = []
    pop = population(pop_size, k1, k2, p1, a, b, x)
    for i in range(n_steps):
        pop.update()
        prop = pop.proportion()
        plt.pie([prop, 1-prop], labels=["Strategy 1", "Strategy 2"])
        plt.title(f'Step {i+1}')
        
        # Save to BytesIO object and encode as base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.read()).decode()
        images.append(img_b64)
        plt.clf()
        
    return jsonify({"images": images})

if __name__ == "__main__":
    app.run(debug=True)
