# A custom autograder for this project

################################################################################
# A mini-framework for autograding
################################################################################

import optparse
import pickle
import random
import sys
import traceback

class WritableNull:
    def write(self, string):
        pass

    def flush(self):
        pass

class Tracker(object):
    def __init__(self, questions, maxes, prereqs, mute_output):
        self.questions = questions
        self.maxes = maxes
        self.prereqs = prereqs

        self.points = {q: 0 for q in self.questions}

        self.current_question = None

        self.current_test = None
        self.points_at_test_start = None
        self.possible_points_remaining = None

        self.mute_output = mute_output
        self.original_stdout = None
        self.muted = False

    def mute(self):
        if self.muted:
            return

        self.muted = True
        self.original_stdout = sys.stdout
        sys.stdout = WritableNull()

    def unmute(self):
        if not self.muted:
            return

        self.muted = False
        sys.stdout = self.original_stdout

    def begin_q(self, q):
        assert q in self.questions
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for prereq in sorted(self.prereqs[q]):
            if self.points[prereq] < self.maxes[prereq]:
                print("""*** NOTE: Make sure to complete Question {} before working on Question {},
*** because Question {} builds upon your answer for Question {}.
""".format(prereq, q, q, prereq))
                return False

        self.current_question = q
        self.possible_points_remaining = self.maxes[q]
        return True

    def begin_test(self, test_name):
        self.current_test = test_name
        self.points_at_test_start = self.points[self.current_question]
        print("*** {}) {}".format(self.current_question, self.current_test))
        if self.mute_output:
            self.mute()

    def end_test(self, pts):
        if self.mute_output:
            self.unmute()
        self.possible_points_remaining -= pts
        if self.points[self.current_question] == self.points_at_test_start + pts:
            print("*** PASS: {}".format(self.current_test))
        elif self.points[self.current_question] == self.points_at_test_start:
            print("*** FAIL")

        self.current_test = None
        self.points_at_test_start = None

    def end_q(self):
        assert self.current_question is not None
        assert self.possible_points_remaining == 0
        print('\n### Question {}: {}/{} ###'.format(
            self.current_question,
            self.points[self.current_question],
            self.maxes[self.current_question]))

        self.current_question = None
        self.possible_points_remaining = None

    def finalize(self):
        import time
        print('\nFinished at %d:%02d:%02d' % time.localtime()[3:6])
        print("\nProvisional grades\n==================")

        for q in self.questions:
          print('Question %s: %d/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %d/%d' % (sum(self.points.values()),
            sum([self.maxes[q] for q in self.questions])))

        print("""
Your grades are NOT yet registered.  To register your grades, make sure
to follow your instructor's guidelines to receive credit on your project.
""")

    def add_points(self, pts):
        self.points[self.current_question] += pts

TESTS = []
PREREQS = {}
def add_prereq(q, pre):
    if isinstance(pre, str):
        pre = [pre]

    if q not in PREREQS:
        PREREQS[q] = set()
    PREREQS[q] |= set(pre)

def test(q, points):
    def deco(fn):
        TESTS.append((q, points, fn))
        return fn
    return deco

def parse_options(argv):
    parser = optparse.OptionParser(description = 'Run public tests on student code')
    parser.set_defaults(
        edx_output=False,
        gs_output=False,
        no_graphics=False,
        mute_output=False,
        check_dependencies=False,
        )
    parser.add_option('--edx-output',
                        dest = 'edx_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_option('--gradescope-output',
                        dest = 'gs_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_option('--question', '-q',
                        dest = 'grade_question',
                        default = None,
                        help = 'Grade only one question (e.g. `-q q1`)')
    parser.add_option('--no-graphics',
                        dest = 'no_graphics',
                        action = 'store_true',
                        help = 'Do not display graphics (visualizing your implementation is highly recommended for debugging).')
    parser.add_option('--mute',
                        dest = 'mute_output',
                        action = 'store_true',
                        help = 'Mute output from executing tests')
    parser.add_option('--check-dependencies',
                        dest = 'check_dependencies',
                        action = 'store_true',
                        help = 'check that numpy and matplotlib are installed')
    (options, args) = parser.parse_args(argv)
    return options

def main():
    options = parse_options(sys.argv)
    if options.check_dependencies:
        check_dependencies()
        return

    if options.no_graphics:
        disable_graphics()

    questions = set()
    maxes = {}
    for q, points, fn in TESTS:
        questions.add(q)
        maxes[q] = maxes.get(q, 0) + points
        if q not in PREREQS:
            PREREQS[q] = set()

    questions = list(sorted(questions))
    if options.grade_question:
        if options.grade_question not in questions:
            print("ERROR: question {} does not exist".format(options.grade_question))
            sys.exit(1)
        else:
            questions = [options.grade_question]
            PREREQS[options.grade_question] = set()

    tracker = Tracker(questions, maxes, PREREQS, options.mute_output)
    for q in questions:
        started = tracker.begin_q(q)
        if not started:
            continue

        for testq, points, fn in TESTS:
            if testq != q:
                continue
            tracker.begin_test(fn.__name__)
            try:
                fn(tracker)
            except KeyboardInterrupt:
                tracker.unmute()
                print("\n\nCaught KeyboardInterrupt: aborting autograder")
                tracker.finalize()
                print("\n[autograder was interrupted before finishing]")
                sys.exit(1)
            except:
                tracker.unmute()
                print(traceback.format_exc())
            tracker.end_test(points)
        tracker.end_q()
    tracker.finalize()

################################################################################
# Tests begin here
################################################################################

import numpy as np
import matplotlib
import contextlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as dset
import torchvision.transforms as T

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import backend

def check_dependencies():
    import matplotlib.pyplot as plt
    import torch
    import time
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    line, = ax.plot([], [], color="black")
    plt.show(block=False)

    for t in range(400):
        angle = t * 0.05
        angle_torch = torch.FloatTensor([angle])
        x = np.sin(angle)
        y = torch.cos(angle_torch).item()
        line.set_data([x,-x], [y,-y])
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(1e-3)

def disable_graphics():
    backend.use_graphics = False

@contextlib.contextmanager
def no_graphics():
    old_use_graphics = backend.use_graphics
    backend.use_graphics = False
    yield
    backend.use_graphics = old_use_graphics

def verify_node(node, expected_type, expected_shape, method_name):
    if expected_type == 'parameter':
        assert node is not None, (
            "{} should return an instance of nn.Parameter, not None".format(method_name))
        assert isinstance(node, nn.Parameter), (
            "{} should return an instance of nn.Parameter, instead got type {!r}".format(
            method_name, type(node).__name__))
    elif expected_type == 'loss':
        assert node is not None, (
            "{} should return an instance a loss node, not None".format(method_name))
        assert isinstance(node, (nn.MSELoss, nn.CrossEntropyLoss)), (
            "{} should return a loss node, instead got type {!r}".format(
            method_name, type(node).__name__))
    elif expected_type == 'node':
        assert node is not None, (
            "{} should return a torch.Tensor object, not None".format(method_name))
        assert isinstance(node, torch.Tensor), (
            "{} should return a torch.Tensor object, instead got type {!r}".format(
            method_name, type(node).__name__))
    else:
        assert False, "If you see this message, please report a bug in the autograder"

    if expected_type != 'loss':
        assert all([(expected is '?' or actual == expected) for (actual, expected) in zip(node.data.size(), expected_shape)]), (
            "{} should return an object with shape {}, got {}".format(
                method_name, expected_shape, tuple(node.data.size())))
                
def verify_model(model, expected_num_weights, model_name):
    total_num_weights = 0
    for kk in model.__dir__():
        attr = getattr(model, kk)
        if isinstance(attr, torch.Tensor):
            total_num_weights += attr.numel()
            print ('Tensor', kk, attr.numel())
        elif isinstance(attr, nn.Module):
            for k, v in attr.state_dict().items():
                print ('Parameter', kk, k, v.numel())
                total_num_weights += v.numel()
    assert np.isclose(total_num_weights, expected_num_weights, atol=115), (
            "The network structure in {} is not the expected. Total number of weights in the expected structure is {}, got {}".format(
                model_name, expected_num_weights, total_num_weights))
    print ("Total model parameter size: {}".format(total_num_weights))
    
@test('q1', points=3)
def check_digit_classification(tracker):
    import models
    model = models.DigitClassificationModel().to(device)
    transform = T.Compose([
               T.ToTensor(),
               T.Normalize((0.1307,), (0.3081,))
            ])
    data_train = dset.MNIST('./data/mnist', train=True, download=True,
                       transform=transform)
    data_test = dset.MNIST('./data/mnist', train=False, download=True,
                       transform=transform)
    loader_test = DataLoader(data_test, batch_size=100)
                       
    detected_parameters = None
    for batch_size in (1, 2, 4):
        loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        inp_x, inp_y = loader_train.__iter__().next()
        inp_x, inp_y = inp_x.to(device), inp_y.to(device)
        output_node = model.forward(inp_x)
        verify_node(output_node, 'node', (batch_size, 10), "DigitClassificationModel.forward()")
    
    verify_model(model, 829145, "DigitClassificationModel")
    tracker.add_points(1) # Partial credit for passing sanity checks

    model.train_model(data_train)

    test_correct = 0.
    test_total = 0.
    x_imgs = []
    y_preds = []
    y_labels = []
    with torch.no_grad():
        model.eval()
        for data, target in loader_test:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += data.size(0)

            x_imgs.append(torch.clamp(data*0.3081+0.1307, 0, 1).detach().cpu())
            y_preds.append(nn.functional.softmax(output, dim=-1).detach().cpu())
            y_labels.append(target.detach().cpu())
        
    x_imgs = torch.cat(x_imgs)
    y_preds = torch.cat(y_preds)
    y_labels = torch.cat(y_labels)
        
    backend.plot_digit_prediction('q1', x_imgs, y_preds, y_labels)
            
    test_accuracy = test_correct / test_total

    accuracy_threshold = 0.98
    if test_accuracy >= accuracy_threshold:
        print("Your final test set accuracy is: {:.2%}".format(test_accuracy))
        tracker.add_points(2)
    else:
        print("Your final test set accuracy ({:.2%}) must be at least {:.0%} to receive full points for this question".format(test_accuracy, accuracy_threshold))

@test('q2', points=3)
def check_regression(tracker):
    import models
    model = models.RegressionModel().to(device)
    
    x = torch.linspace(-2 * np.pi, 2 * np.pi, 2048).view(-1, 1) # shape (2048, 1)
    y = torch.sin(x) # shape (2048, 1)
    data_train = TensorDataset(x, y)
    
    x = torch.linspace(-2 * np.pi, 2 * np.pi, 200).view(-1, 1)
    y = torch.sin(x)
    data_val = TensorDataset(x, y)
    
    x = torch.linspace(-2 * np.pi, 2 * np.pi, 257).view(-1, 1)
    y = torch.sin(x)
    data_test = TensorDataset(x, y)
    
    detected_parameters = None
    for batch_size in (1, 2, 4):
        inp_x = x[:batch_size]
        inp_y = y[:batch_size]
        inp_x = inp_x.to(device)
        output_node = model.forward(inp_x)
        verify_node(output_node, 'node', (batch_size, 1), "RegressionModel.forward()")

    tracker.add_points(1) # Partial credit for passing sanity checks

    model.train_model(data_train, data_val)
    backend.maybe_sleep_and_close(1)

    # Re-compute the loss ourselves: otherwise get_loss() could be hard-coded
    # to always return zero
    model.eval()
    test_predicted = model.forward(data_test.tensors[0].to(device))
    verify_node(test_predicted, 'node', (data_test.tensors[0].shape[0], 1), "RegressionModel.forward()")
    sanity_loss = 0.5 * torch.mean((test_predicted - data_test.tensors[1].to(device))**2).item()
    
    backend.plot_regression(data_test.tensors[0].squeeze(), data_test.tensors[1].squeeze(), test_predicted.squeeze())

    loss_threshold = 0.02
    if sanity_loss <= loss_threshold:
        print("Your final loss is: {:f}".format(sanity_loss))
        tracker.add_points(2)
    else:
        print("Your final loss ({:f}) must be no more than {:.4f} to receive full points for this question".format(sanity_loss, loss_threshold))
        
@test('q3', points=2)
def check_adversarial_examples(tracker):
    import models
    model = models.DigitAttackModel()
    
    data_test = dset.MNIST('./data/mnist', train=False, download=True,
                       transform=T.ToTensor())
    loader_test = DataLoader(data_test, batch_size=100)
    epsilon = 0.2

    detected_parameters = None
    for batch_size in (1, 2, 4):
        loader_train = DataLoader(data_test, batch_size=batch_size, shuffle=True)
        inp_x, inp_y = loader_train.__iter__().next()
        inp_x, inp_y = inp_x.to(device), inp_y.to(device)
        output_node = model.attack(inp_x, inp_y, epsilon)
        verify_node(output_node, 'node', (batch_size, 1, 28, 28), "DigitAttackModel.attack()")
    
    test_correct = 0.
    test_total = 0.
    x_imgs = []
    y_preds = []
    y_labels = []
    for batch_idx, (data, target) in enumerate(loader_test):
        data, target = data.to(device), target.to(device)
        x_adv = model.attack(data, target, epsilon)
        assert x_adv.min().item() > -0.001 and x_adv.max().item() < 1.001, (
                "The elements of adversarial examples must be in the interval [0,1]")
        assert torch.abs(x_adv - data).max() < epsilon + 0.001, (
                "Adversarial examples must be close to the original data.")
        output = model.model((x_adv-0.1307) / 0.3081)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        test_correct += pred.eq(target.view_as(pred)).sum().item()
        test_total += data.size(0)
        
        if batch_idx % 20 == 0:
            print('Execute [{}/{} ({:.0f}%)]'.format(batch_idx, len(loader_test),
                        100. * batch_idx / len(loader_test)))
        
        x_imgs.append(x_adv.detach().cpu())
        y_preds.append(nn.functional.softmax(output, dim=-1).detach().cpu())
        y_labels.append(target.detach().cpu())
        
    test_accuracy = test_correct / test_total
    
    x_imgs = torch.cat(x_imgs)
    y_preds = torch.cat(y_preds)
    y_labels = torch.cat(y_labels)
        
    backend.plot_digit_prediction('q3', x_imgs, y_preds, y_labels)

    accuracy_threshold = 0.50
    if test_accuracy <= accuracy_threshold:
        print("Your final test set accuracy is: {:.2%}".format(test_accuracy))
        tracker.add_points(2)
    else:
        print("Your final test set accuracy ({:.2%}) must be at most {:.0%} to receive full points for this question".format(test_accuracy, accuracy_threshold))
        
@test('q4', points=4)
def check_lang_id(tracker):
    import models
    batch_size = 64
    model = models.LanguageIDModel().to(device)
    loader_train = backend.LanguageIDDataLoader(64, 'train')
    loader_dev = backend.LanguageIDDataLoader(64, 'dev')
    loader_test = backend.LanguageIDDataLoader(64, 'test')

    detected_parameters = None
    dataset = loader_train
    for batch_size, word_length in ((1, 1), (2, 1), (2, 6), (4, 8)):
        start = dataset.dev_buckets[-1, 0]
        end = start + batch_size
        inp_xs, inp_y = dataset._encode(dataset.dev_x[start:end], dataset.dev_y[start:end])
        inp_xs = inp_xs[:word_length]

        output_node = model.forward(inp_xs)
        verify_node(output_node, 'node', (batch_size, len(dataset.language_names)), "LanguageIDModel.forward()")

    tracker.add_points(1) # Partial credit for passing sanity checks

    model.train_model(loader_train, loader_dev)
    
    _, test_accuracy = backend.get_loss_and_accuracy(model, loader_test, device)
    accuracy_threshold = 0.81
    accuracy_threshold_1 = 0.60
    accuracy_threshold_2 = 0.75
    if test_accuracy >= accuracy_threshold:
        print("Your final test set accuracy is: {:.2%}".format(test_accuracy))
        tracker.add_points(3)
    else:
        print("Your final test set accuracy ({:.2%}) must be at least {:.0%} to receive full points for this question".format(test_accuracy, accuracy_threshold))
        if test_accuracy >= accuracy_threshold_1:
            tracker.add_points(1)
        if test_accuracy >= accuracy_threshold_2:
            tracker.add_points(1)

@test('q5', points=3)
def check_rl(tracker):
    import models, backend

    num_trials = 6
    trials_satisfied = 0
    trials_satisfied_small = 0
    trials_satisfied_required = 3
    for trial_number in range(num_trials):
        model = models.DeepQModel()
        model.train_model()

        stats = model.data_loader.stats
        if stats['mean_reward'] >= stats['reward_threshold']:
            trials_satisfied += 1
        if stats['mean_reward'] >= stats['reward_threshold_small']:
            trials_satisfied_small += 1

        if trials_satisfied >= trials_satisfied_required:
            tracker.add_points(3)
            return
        else:
            trials_left = num_trials - (trial_number + 1)
            if trials_satisfied_small + trials_left < trials_satisfied_required:
                break
    if trials_satisfied_small >= trials_satisfied_required:
        tracker.add_points(1)
    print(
        "To receive credit for this question, your agent must receive a mean reward of at least {} on {} out of {} trials".format(
            stats['reward_threshold'], trials_satisfied_required, num_trials))


    
if __name__ == '__main__':
    main()
