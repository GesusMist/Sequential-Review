import openreview
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.stats import norm
from collections import Counter
import collections
import itertools
from math import comb
import json
from json import JSONDecodeError
from datetime import datetime

def extract_number(string):
    colon_index = string.find(":")
    if colon_index != -1:
        number_str = string[:colon_index].strip()
        if number_str.isdigit():
            return int(number_str)

client = openreview.Client(baseurl='https://api.openreview.net')

    
submissions = client.get_all_notes(
    invitation="ICLR.cc/2023/Conference/-/Blind_Submission",
    details='directReplies'
)

papers = []
for submission in submissions:
    authorids = submission.content['authorids']
    scores = []
    for reply in submission.details["directReplies"]:
        if reply["invitation"].endswith("Decision"):
            forum = reply['forum']
            decision = reply['content']['decision']
        if reply["invitation"].endswith("Official_Review"):
            score = []
            for val_str in reply['content'].values():
                if (isinstance(val_str, str) and val_str.split(':')[0].isnumeric()):
                  score.append(int((val_str.split(':'))[0]))
            scores += [score[-1]]
    review = {'forum': forum, 'authorids': authorids, 'decision': decision, 'scores': scores}
    papers.append(review)