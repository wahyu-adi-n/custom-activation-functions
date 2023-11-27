import os
import matplotlib.pyplot as plt

# Set counters
n_samples_nr_train = len(os.listdir('chest_xray/train/NORMAL'))
n_samples_pn_train = len(os.listdir('chest_xray/train/PNEUMONIA'))

n_samples_nr_val = len(os.listdir('chest_xray/val/NORMAL'))
n_samples_pn_val = len(os.listdir('chest_xray/val/PNEUMONIA'))

n_samples_nr_test = len(os.listdir('chest_xray/test/NORMAL'))
n_samples_pn_test = len(os.listdir('chest_xray/test/PNEUMONIA'))

# Define two dictionaries
class_names = {0: 'normal', 
               1: 'pneumonia'}

class_count = {0: n_samples_nr_train + n_samples_nr_val + n_samples_nr_test, 
               1: n_samples_pn_train + n_samples_pn_val + n_samples_pn_test}

# Print results
print(f'Found {class_count[0]} elements for {class_names[0]}')
print(f'Found {class_count[1]} elements for {class_names[1]}')

# Barplot
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(class_names.values(), class_count.values(), color=['skyblue', 'orange'])
ax.set_title("Pneumonia Dataset Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Frequency")

# Annotate the bars with counts
for i, v in enumerate(class_count.values()):
    ax.text(i, v + 20, str(v), ha='center', va='bottom')

plt.savefig("distribution_plot.png", format='png')
plt.show()
