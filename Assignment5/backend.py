import collections
import os
import time
import random
import math
from collections import deque, namedtuple, defaultdict

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, sampler

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
use_graphics = True

def maybe_sleep_and_close(seconds):
    if use_graphics and plt.get_fignums():
        time.sleep(seconds)
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            plt.close(fig)
            try:
                # This raises a TclError on some Windows machines
                fig.canvas.start_event_loop(1e-3)
            except:
                pass

def get_data_path(filename):
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise Exception("Could not find data file: {}".format(filename))
    return path

def get_loss_and_accuracy(model, data_loader, device=None):
    num_correct = 0.
    num_total = 0.
    loss_total = 0.
    flag = model.training # store the original train/eval mode
    model.eval()
    for x, y in data_loader:
        if device is not None:
            x, y = x.to(device), y.to(device)
        output = model(x)
        loss_total += F.cross_entropy(output, y).item()
        num_correct += torch.sum(torch.max(output, 1)[1] == y).item()
        num_total += len(y)

    model.training = flag # restore the original train/eval mode
    return loss_total/num_total, num_correct/num_total
    
def plot_regression(x, y, prediction):
    plt.cla()
    plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy(), label='sin(x)')
    plt.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'r-', lw=5, label='predicted')
    plt.legend()
    plt.savefig('figures/q2.png')
    
def plot_digit_prediction(question, dev_images, dev_probs, dev_labels):
    width = 20  # Width of each row expressed as a multiple of image width
    samples = 100  # Number of images to display per label
    fig = plt.figure(figsize=(10,8))
    ax = {}
    images = defaultdict(list)
    texts = defaultdict(list)
    for i in reversed(range(10)):
        ax[i] = plt.subplot2grid((30, 1), (3 * i, 0), 2, 1, sharex=ax.get(9))
        plt.setp(ax[i].get_xticklabels(), visible=i == 9)
        ax[i].set_yticks([])
        ax[i].text(-0.03, 0.5, i, transform=ax[i].transAxes, va="center")
        ax[i].set_xlim(0, 28 * width)
        ax[i].set_ylim(0, 28)
        for j in range(samples):
            images[i].append(ax[i].imshow(
                np.zeros((28, 28)), vmin=0, vmax=1, cmap="Greens", alpha=0.3))
            texts[i].append(ax[i].text(
                0, 0, "", ha="center", va="top", fontsize="smaller"))
    ax[9].set_xticks(np.linspace(0, 28 * width, 11))
    ax[9].set_xticklabels(["{:.1f}".format(num) for num in np.linspace(0, 1, 11)])
    ax[9].tick_params(axis="x", pad=16)
    ax[9].set_xlabel("Probability of Correct Label")
    status = ax[0].text(0.5, 1.5, "", transform=ax[0].transAxes, ha="center", va="bottom")
    
    dev_images = dev_images.detach().cpu().numpy()
    dev_probs = dev_probs.detach().cpu().numpy()
    dev_predicted = np.argmax(dev_probs, axis=1)
    dev_labels = dev_labels.detach().cpu().numpy()
    dev_accuracy = np.mean(dev_predicted == dev_labels)

    status.set_text("accuracy: {:.2%}".format(dev_accuracy))
    for i in range(10):
        predicted = dev_predicted[dev_labels == i]
        probs = dev_probs[dev_labels == i][:, i]
        linspace = np.linspace(0, len(probs) - 1, samples).astype(int)
        indices = probs.argsort()[linspace]
        for j, (prob, image) in enumerate(zip(probs[indices], dev_images[dev_labels == i][indices])):
            images[i][j].set_data(image.reshape((28, 28)))
            left = prob * (width - 1) * 28
            if predicted[indices[j]] == i:
                images[i][j].set_cmap("Greens")
                texts[i][j].set_text("")
            else:
                images[i][j].set_cmap("Reds")
                texts[i][j].set_text(predicted[indices[j]])
                texts[i][j].set_x(left + 14)
            images[i][j].set_extent([left, left + 28, 0, 28])
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(1e-3)
    
    plt.savefig('figures/{}.png'.format(question))

class LanguageIDDataLoader(object):
    def __init__(self, batch_size, phase='train'):
        data_path = get_data_path("lang_id.npz")
        self.phase = phase
        self.batch_size = batch_size

        with np.load(data_path) as data:
            self.chars = data['chars']
            self.language_codes = data['language_codes']
            self.language_names = data['language_names']

            self.train_x = data['train_x']
            self.train_y = data['train_y']
            self.train_buckets = data['train_buckets']
            self.dev_x = data['dev_x']
            self.dev_y = data['dev_y']
            self.dev_buckets = data['dev_buckets']
            self.test_x = data['test_x']
            self.test_y = data['test_y']
            self.test_buckets = data['test_buckets']
            
        assert isinstance(batch_size, int) and batch_size > 0, (
            "Batch size should be a positive integer, got {!r}".format(
                batch_size))
        assert self.train_x.shape[0] >= batch_size, (
            "Dataset size {:d} is smaller than the batch size {:d}".format(
                self.train_x.shape[0], batch_size))
        # self.length = {}
        # self.length['train'] = self.train_x.shape[0] // batch_size
        # self.length['dev'] = sum(int(np.ceil((end-start)/batch_size)+0.5) for start, end in self.dev_buckets)
        # self.length['test'] = sum(int(np.ceil((end-start)/batch_size)+0.5) for start, end in self.test_buckets)
        
        self.bucket_weights = self.train_buckets[:,1] - self.train_buckets[:,0]
        self.bucket_weights = self.bucket_weights / float(self.bucket_weights.sum())

        self.chars_print = self.chars
        try:
            print(u"Alphabet: {}".format(u"".join(self.chars)))
        except UnicodeEncodeError:
            self.chars_print = "abcdefghijklmnopqrstuvwxyzaaeeeeiinoouuacelnszz"
            print("Alphabet: " + self.chars_print)
            self.chars_print = list(self.chars_print)
            print("""
NOTE: Your terminal does not appear to support printing Unicode characters.
For the purposes of printing to the terminal, some of the letters in the
alphabet above have been substituted with ASCII symbols.""".strip())
        print("")

        # Select some examples to spotlight in the monitoring phase (3 per language)
        spotlight_idxs = []
        for i in range(len(self.language_names)):
            idxs_lang_i = np.nonzero(self.dev_y == i)[0]
            idxs_lang_i = np.random.choice(idxs_lang_i, size=3, replace=False)
            spotlight_idxs.extend(list(idxs_lang_i))
        self.spotlight_idxs = np.array(spotlight_idxs, dtype=int)

        # Templates for printing updates as training progresses
        max_word_len = self.dev_x.shape[1]
        max_lang_len = max([len(x) for x in self.language_names])

        self.predicted_template = u"Pred: {:<NUM}".replace('NUM',
            str(max_lang_len))

        self.word_template = u"  "
        self.word_template += u"{:<NUM} ".replace('NUM', str(max_word_len))
        self.word_template += u"{:<NUM} ({:6.1%})".replace('NUM', str(max_lang_len))
        self.word_template += u" {:<NUM} ".replace('NUM',
            str(max_lang_len + len('Pred: ')))
        for i in range(len(self.language_names)):
            self.word_template += u"|{}".format(self.language_codes[i])
            self.word_template += "{probs[" + str(i) + "]:4.0%}"

        self.last_update = time.time()

    def _encode(self, inp_x, inp_y):
        xs = []
        for i in range(inp_x.shape[1]):
            if np.all(inp_x[:,i] == -1):
                break
            assert not np.any(inp_x[:,i] == -1), (
                "Please report this error in the project: batching by length was done incorrectly in the provided code")
            x = np.eye(len(self.chars))[inp_x[:,i]]
            xs.append(x)
        xs = torch.from_numpy(np.stack(xs)).float()
        y = torch.from_numpy(inp_y).long()
        return xs, y
        
    def __iter__(self):
        if self.phase == 'train':
            batch_size = self.batch_size
            for iteration in range(self.train_x.shape[0] // batch_size):
                bucket_id = np.random.choice(self.bucket_weights.shape[0], p=self.bucket_weights)
                example_ids = self.train_buckets[bucket_id, 0] + np.random.choice(
                    self.train_buckets[bucket_id, 1] - self.train_buckets[bucket_id, 0],
                    size=batch_size)

                yield self._encode(self.train_x[example_ids], self.train_y[example_ids])
        
        elif self.phase == 'dev':
            for start, end in self.dev_buckets:
                for i in range(start, end, self.batch_size):
                    i_end = min(i+self.batch_size, end)
                    yield self._encode(self.dev_x[i:i_end], self.dev_y[i:i_end])

        elif self.phase == 'test':
            for start, end in self.test_buckets:
                for i in range(start, end, self.batch_size):
                    i_end = min(i+self.batch_size, end)
                    yield self._encode(self.test_x[i:i_end], self.test_y[i:i_end])
        
    def __len__(self):
        batch_size = self.batch_size
        if self.phase == 'train':
            return self.train_x.shape[0] // batch_size
        elif self.phase == 'dev':
            return sum(int(np.ceil((end-start)/batch_size)+0.5) for start, end in self.dev_buckets)
        elif self.phase == 'test':
            return sum(int(np.ceil((end-start)/batch_size)+0.5) for start, end in self.test_buckets)


class CartPoleEnv(object):
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    # Licensed under MIT license: https://opensource.org/licenses/MIT

    def __init__(self, theta_threshold_degrees=12, seed=1, max_steps=200):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.max_steps = max_steps

        # Angle at which to fail the episode
        self.theta_threshold_degrees = theta_threshold_degrees
        self.theta_threshold_radians = theta_threshold_degrees * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = {0, 1}
        self.num_actions = len(self.action_space)
        self.observation_state_size = 2

        self.np_random = np.random.RandomState(seed)
        self.state = None

        self.steps_taken = 0
        self.steps_beyond_done = None

    def reset(self):
        self.steps_taken = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def step(self, action):
        assert action in self.action_space, "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        done = (x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians)
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:  # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                print("You are calling 'step()' even though this environment "
                      "has already returned done = True. You should always "
                      "call 'reset()' once you receive 'done = True' -- any "
                      "further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        self.steps_taken += 1

        if self.steps_taken >= self.max_steps:
            done = True

        return np.array(self.state), reward, done, {}


Transition = namedtuple("Transition", field_names=[
    "state", "action", "reward", "next_state", "done"])


class ReplayMemory(object):
    def __init__(self, capacity):
        """Replay memory class

        Args:
            capacity (int): Max size of this memory
        """
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        """Creates `Transition` and insert

        Args:
            state (np.ndarray): 1-D tensor of shape (input_dim,)
            action (int): action index (0 <= action < output_dim)
            reward (int): reward value
            next_state (np.ndarray): 1-D tensor of shape (input_dim,)
            done (bool): whether this state was last step
        """
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = Transition(state, action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size):
        """Returns a minibatch of `Transition` randomly

        Args:
            batch_size (int): Size of mini-bach

        Returns:
            List[Transition]: Minibatch of `Transition`
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the length """
        return len(self.memory)


class CartPoleLoader(object):
    def __init__(self, model):
        self.model = model
        self.n_episode = 200
        self.stats = {}
        
    def __iter__(self):
        # Adapted from https://gist.github.com/kkweon/52ea1e118101eb574b2a83b933851379
        self.stats = {}
        self.stats['mean_reward'] = 0

        # Max size of the replay buffer
        capacity = 50000

        # After max episode, eps will be `min_eps`
        max_eps_episode = 50

        # eps will never go below this value
        min_eps = 0.01

        # Number of transition samples in each minibatch update
        batch_size = 64

        # Number of episodes between rendering the environment
        play_every = 10

        # Discount parameter
        gamma = 0.95

        # Max number of episodes to run
        n_episode = self.n_episode

        # Failure if the pole falls past this many degrees
        theta_threshold_degrees = 60

        # Random seed
        seed = 1

        # Win if you average at least this much reward (max reward is 200) for
        # num_episodes_to_average consecutive episodes
        reward_threshold = 180
        reward_threshold_small = 100
        num_episodes_to_average = 10

        # If set (an integer), clip the absolute difference between Q_pred and
        # Q_target to be no more than this
        td_error_clipping = None

        episode_print_interval = 10

        self.stats['reward_threshold'] = reward_threshold
        self.stats['reward_threshold_small'] = reward_threshold_small    

        env = CartPoleEnv(theta_threshold_degrees, seed=seed)
        rewards = deque(maxlen=num_episodes_to_average)
        input_dim, output_dim = env.observation_state_size, env.num_actions
        replay_memory = ReplayMemory(capacity)

        cart_width = 1.0
        cart_height = 0.1
        pole_width = 0.05
        pole_height = 2.0

        def get_cart_coords(x):
            return [
                (x - cart_width / 2, -cart_height),
                (x + cart_width / 2, -cart_height),
                (x + cart_width / 2, cart_height),
                (x - cart_width / 2, cart_height),
            ]

        def get_pole_coords(x, theta):
            bottom_left = np.array([
                x + pole_width * np.cos(np.pi - theta),
                pole_width * np.sin(np.pi - theta)])
            bottom_right = np.array([
                x + pole_width * np.cos(-theta),
                pole_width * np.sin(-theta)])
            top_offset = np.array([
                pole_height * np.cos(np.pi / 2 - theta),
                pole_height * np.sin(np.pi / 2 - theta)])
            return [
                bottom_left,
                bottom_right,
                bottom_right + top_offset,
                bottom_left + top_offset
            ]

        if use_graphics:
            import matplotlib.patches as patches
            fig, ax = plt.subplots(1, 1)
            ax.set_xlim(-env.x_threshold - cart_width, env.x_threshold + cart_width)
            ax.set_ylim(-cart_height / 2, pole_height * 1.1)
            ax.set_aspect("equal")
            cart_polygon = patches.Polygon(get_cart_coords(0), color="black")
            pole_polygon = patches.Polygon(get_pole_coords(0, 0), color="blue")
            ax.add_patch(pole_polygon)
            ax.add_patch(cart_polygon)
            text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
            plt.show(block=False)
            
        def getQValue(model, states, device=None):
            states = torch.from_numpy(states).float()
            if device is not None:
                states = states.to(device)
            q = model.forward(states)
            return q.detach().cpu().numpy()

        def train_helper(minibatch):
            """Prepare minibatches

            Args:
                minibatch (List[Transition]): Minibatch of `Transition`

            Returns:
                float: Loss value
            """
            states = np.vstack([x.state for x in minibatch])
            actions = np.array([x.action for x in minibatch])
            rewards = np.array([x.reward for x in minibatch])
            next_states = np.vstack([x.next_state for x in minibatch])
            done = np.array([x.done for x in minibatch])

            Q_predict = getQValue(self.model, states, device)
            Q_target = np.copy(Q_predict)
            Q_target[np.arange(len(Q_target)), actions] = (
                    rewards + gamma * np.max(getQValue(self.model, next_states, device), axis=1) * ~done)

            if td_error_clipping is not None:
                Q_target = Q_predict + np.clip(
                    Q_target - Q_predict, -td_error_clipping, td_error_clipping)

            # print("max target", Q_target.max())
            # print("max error", np.abs(error).max())

            return Q_predict, Q_target

        annealing_slope = (min_eps - 1.0) / max_eps_episode

        for episode in range(n_episode):
            eps = max(annealing_slope * episode + 1.0, min_eps)
            render = play_every != 0 and (episode + 1) % play_every == 0

            s = env.reset()
            done = False
            total_reward = 0

            while not done:
                a = self.model.get_action(torch.from_numpy(s[np.newaxis, :]).float(), eps)
                s2, r, done, info = env.step(a)

                total_reward += r

                if render and use_graphics:
                    x, x_dot, theta, theta_dot = env.state
                    cart_polygon.set_xy(get_cart_coords(x))
                    pole_polygon.set_xy(get_pole_coords(x, theta))
                    text.set_text("episode: {:,}/{:,}\nreward: {}".format(
                        episode + 1, n_episode, total_reward))
                    fig.canvas.draw_idle()
                    fig.canvas.start_event_loop(1e-3)

                replay_memory.push(s, a, r if not done else -1, s2, done)

                if len(replay_memory) > batch_size:
                    minibatch = replay_memory.pop(batch_size)
                    Q_predict, Q_target = train_helper(minibatch)
                    states = np.vstack([x.state for x in minibatch])
                    yield torch.from_numpy(states).float(), torch.from_numpy(Q_target).float()

                s = s2

            rewards.append(total_reward)
            if (episode + 1) % episode_print_interval == 0:
                print("[Episode: {:3}] Reward: {:5} Mean Reward of last {} episodes: {:5.1f} epsilon: {:5.2f}".format(
                    episode + 1, total_reward, num_episodes_to_average, np.mean(rewards), eps))

            if len(rewards) == rewards.maxlen:
                self.stats['mean_reward'] = np.mean(rewards)
                if np.mean(rewards) >= reward_threshold:
                    print("Completed in {} episodes with mean reward {}".format(
                        episode + 1, np.mean(rewards)))
                    self.stats['reward_threshold_met'] = True
                    break
        else:
            # reward threshold not met
            print("Aborted after {} episodes with mean reward {}".format(
                episode + 1, np.mean(rewards)))

        if use_graphics:
            plt.close(fig)
            try:
                # This raises a TclError on some Windows machines
                fig.canvas.start_event_loop(1e-3)
            except:
                pass

