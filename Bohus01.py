import torch
import torch.nn.functional as F
import time


class ScoreCalculator:
    def __init__(self, num_hands, num_ranks, num_suits, device, verbose=False):
        self.num_hands = num_hands
        self.num_ranks = num_ranks
        self.num_suits = num_suits
        self.device = device
        self.verbose = verbose
        self.dtype = torch.long

    def calc_score(self, hands):
        if self.verbose:
            print('hands:\n', hands)
        not_zero = torch.clamp(hands, 0, 1)
        if self.verbose:
            print('not_zero:\n', not_zero)
        # padding
        pad_not_zero = F.pad(not_zero, (1, 1), mode='constant', value=0)
        triplet_center = pad_not_zero[:, :-2] * pad_not_zero[:, 1:-1] * pad_not_zero[:, 2:]
        if self.verbose:
            print('triplet_center:\n', triplet_center)
        # padding
        triplet_center = F.pad(triplet_center, (1, 1), mode='constant', value=0)
        alive = (triplet_center[:, :-2] + triplet_center[:, 1:-1] + triplet_center[:, 2:])
        alive = torch.clamp(alive, 0, 1)
        if self.verbose:
            print('alive:\n', alive)
        hands_alive = alive * hands
        if self.verbose:
            print('hands_alive:\n', hands_alive)

        partial_scores = torch.ones(self.num_hands, dtype=self.dtype, device=self.device) * hands_alive[:, 0]
        scores = torch.zeros(self.num_hands, dtype=self.dtype, device=self.device)
        is_zero = 1-not_zero

        for r in range(1, self.num_ranks):
            scores += partial_scores * is_zero[:, r]
            partial_scores = partial_scores * hands_alive[:, r] + hands_alive[:, r] * is_zero[:, r-1]

        scores += partial_scores
        return scores, partial_scores


def generate_hands(bas, n_r):
    n_h = bas ** n_r
    indices = torch.arange(n_h, dtype=torch.long)
    han = torch.zeros((n_h, n_r), dtype=torch.long)
    for r in range(n_r - 1, -1, -1):
        han[:, r] = indices % bas
        indices //= bas
    return han


start = time.time()
# dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dev = 'cpu'

n_ranks, n_suits = 13, 4
values = torch.clamp(torch.arange(1, n_ranks+1, dtype=torch.long), max=10)

print('values: ', values)
base = n_suits+1
n_ranks1 = 8
n_ranks2 = n_ranks - n_ranks1 - 1
n_hands1 = base ** n_ranks1
n_hands2 = base ** n_ranks2
values1 = values[: n_ranks1]
values2 = torch.flip(values[-n_ranks2:], dims=[0])
x_value = values[n_ranks1]
print('values1: ', values1)
print('values2: ', values2)
print('x_value: ', x_value)

hands1 = generate_hands(base, n_ranks1)
print('hands1.shape: ', hands1.shape)
score_calc = ScoreCalculator(n_hands1, n_ranks1, n_suits, dev, False)
score1, partial1 = score_calc.calc_score(hands1)
value_score1 = torch.matmul(hands1, values1)
print('score1, partial1, value_score1 shapes: ', score1.shape, partial1.shape, value_score1.shape)

hands2 = generate_hands(base, n_ranks2)
print('hands2.shape: ', hands2.shape)
score_calc = ScoreCalculator(n_hands2, n_ranks2, n_suits, dev, False)
score2, partial2 = score_calc.calc_score(hands2)
value_score2 = torch.matmul(hands2, values2)
print('score2, partial2, value_score2 shapes: ', score2.shape, partial2.shape, value_score2.shape)

number_found = 0

for x in range(base):
    print(x)
    value_x = x * x_value
    for i2 in range(n_hands2):
        value_score = value_score1 + value_x + value_score2[i2]
        score = score1 + score2[i2] + x * partial1 * partial2[i2]
        diff_is_zero = 1-torch.clamp(torch.abs(score - value_score), max=1)
        number_found += torch.sum(diff_is_zero)
        a = 42

print('number found: ', number_found)

elapsed_time = (time.time() - start) * 1000.0
print(f"Elapsed time: {elapsed_time:.1f} msec")
# hands_per_msec = num_iter * n_hands / elapsed_time
# print(f"Scored per msec: {hands_per_msec:.0f}")
# print(f"Scored per minute: {hands_per_msec * 1000 * 60 / 1000000:.0f} M")

# print('\nscore = ', score)
