using PyCall

py"""
import torch
import torch.nn as nn
import time
import math

n_positions = 20
def fromOneHot(tensor):
    return torch.argmax(tensor, dim=1)
def posToTensor(pos):
    tensor = torch.zeros(1, n_positions)
    tensor[0][pos - 1] = 1
    return tensor

def exampleToTensor(trajectory):
    obs_trajectory = [state['obs'] for state in trajectory]
    obs_vec_list = [posToTensor(pos) for pos in obs_trajectory]
    return torch.stack(obs_vec_list)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer_size=None, n_hidden_layers=3):
        super(RNN, self).__init__()

        if hidden_layer_size is None:
            hidden_layer_size = input_size + hidden_size

        self.hidden_size = hidden_size

        self.i2l1 = nn.Linear(input_size + hidden_size, hidden_layer_size)
        self.inner_layers = nn.ModuleList([nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(n_hidden_layers - 1)])
        self.l2o = nn.Linear(hidden_layer_size, output_size)
        self.l2h = nn.Linear(hidden_layer_size, hidden_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)

        l = self.i2l1(combined)
        for layer in self.inner_layers:
            l = layer(l)
        hidden = self.l2h(l)
        output = self.softmax(self.l2o(l))

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def runRNNOnExample(rnn, example):
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(example.size()[0] - 1):
        input = example[i]
        output, hidden = rnn(input, hidden)

    return output

criterion = nn.NLLLoss()
learning_rate = 0.001
def train(rnn, example):
    hidden = rnn.initHidden()
    rnn.zero_grad()

    loss = 0
    for i in range(example.size()[0] - 1):
        input = example[i]
        target = example[i + 1]
        output, hidden = rnn(input, hidden)
        loss += criterion(output, fromOneHot(target))

    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / example.size()[0]

def runTrainingLoop(rnn, getRandomTrainingExample, n_iters=100000, print_every=500, plot_every=100):
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()
    for iter in range(1, n_iters + 1):
        example = exampleToTensor(getRandomTrainingExample())
        output, loss = train(rnn, example)
        current_loss += loss

            # Print iter number, loss, name and guess
        if iter % print_every == 0:
            print('%d %d%% (%s) %.4f' % (iter, math.floor(iter / n_iters * 100), timeSince(start), loss))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    return all_losses
"""

function trace_to_object(tr)
    return [
        Dict(
            "x" => latents_choicemap(tr, t)[:xₜ => :val],
            "v" => latents_choicemap(tr, t)[:vₜ => :val],
            "obs" => obs_choicemap(tr, t)[:obs => :val]
        )
        for t=1:get_args(tr)[1]
    ]
end

function get_trained_rnn(params)
    rnn = py"RNN"(length(Positions()), 64, length(Positions()))
    println("Constructed RNN.  Training now...")
    loss_record = py"runTrainingLoop"(
        rnn,
        () -> trace_to_object(simulate(model, (params.n_steps_per_run,))),
        params.training_params...
    )
    println("Training completed.")
    println()
    return (rnn, loss_record)
end

function runRNNOnObsTrace(rnn, obs_tr)
    obs_vec_list = [py"posToTensor"(obj["obs"]) for obj in trace_to_object(obs_tr)]
    obs_vec = py"torch".stack(obs_vec_list)
    return py"runRNNOnExample"(rnn, obs_vec)
end