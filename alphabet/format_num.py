#!/usr/bin/env python3

from sys import argv

output = ''
indent = '\t'*4
with open(argv[1]) as read:
    for line in read:
        a, b = line.split(', ')
        output += f'{indent}[{float(a):.2f}, {float(b):.2f}],\n'

print(f'np.array([\n{output}{indent[:3]}]), ')
