# Rememberer
[original Rememberer paper](https://arxiv.org/abs/2306.07929)

Project created for EECS 598 - Large Language Models, Winter 2024

We use a large language model as a semi-parametric reinforcement learning agent, testing its performance on WebShop, an online web store task. The agent is augmented with an external experience memory which allows it to iteratively refine and improve its own prompt. Relevant experiences are selected based on embedding similarity and template matching. Our best performing model achieves a 36.8% successful completion rate, and a trial is considered a success if the agent is able to accurately select the item along with all necessary category attributes before clicking 'buy now'. 

### Video Demo

https://github.com/oliveraw/rememberer/assets/69375421/0b2da232-55fc-4ba8-b274-da1b1f324a3c

