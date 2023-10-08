from random import random, randint

output_lines = []

with open("data/tcav/sentiment-classification/raw-positive-adjectives.csv") as df:
    for line in df:
        cleaned_line = line.strip()
        op_line = [".", ".", ".", ".", ".", ".", "."]
        random_index = randint(0, 6)
        op_line[random_index] = cleaned_line
        # print(" ".join(op_line))
        output_lines.append(" ".join(op_line))

with open("data/tcav/sentiment-classification/positive-adjectives.csv", "w") as df:
    for op_line in output_lines:
        #print(op_line)
        df.write(op_line + "\n")

