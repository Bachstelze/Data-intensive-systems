import csv

INPUT = "A15/scores.csv"
OUTPUT = "A15/scores_rescaled.csv"

with open(INPUT, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [(row[0], float(row[1])) for row in reader]

# Find the max score (worst exercise in dataset)
scores = [v for _, v in rows]
max_score = max(scores)
print(f"Max score (worst exercise): {max_score}")

# Rescale linearly: new = (old / max_score) * 4
rescaled = [(name, round((val / max_score) * 4, 6)) for name, val in rows]

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Var1", "Var2_rescaled"])
    writer.writerows(rescaled)

print(f"Written {len(rescaled)} rows to {OUTPUT}")

# Show a few examples
print("\nExamples:")
for name, old, new in [(rows[i][0], rows[i][1], rescaled[i][1]) for i in [0, 50, 100, -1]]:
    print(f"  {name}: {old} → {new}")
