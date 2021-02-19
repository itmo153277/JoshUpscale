#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Clean syncing (remove norm spikes)."""

import json
import numpy as np
import pandas

with open("norms.txt", "r") as f:
    norms = f.readlines()
norms = np.array([json.loads(norm) for norm in norms])
means = np.array([x for x in pandas.Series(
    norms).rolling(3, min_periods=1).mean()])
norms[means == 0] = 1
means[means == 0] = 1
bads = np.where(norms / means > 1.2)[0]
for bad in bads[::-1]:
    print("DeleteFrame(%d)" % bad)
