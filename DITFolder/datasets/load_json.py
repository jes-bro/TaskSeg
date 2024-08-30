import json

# Define the template dictionary
template = {
    'train': {
        'refs': "The bag of loose tea leaves.",
    },
    'val': {
        'refs': "The bag of loose tea leaves.",
    },
    'test': {
        'refs': "The bag of loose tea leaves.",
    },
}

# Create a dictionary with keys 'train', 'val', and 'test' each containing a list of references
stat_refs_dict = {
    'train': [template['train'] for _ in range(107)],
    'val': [template['val'] for _ in range(107)],
    'test': [template['test'] for _ in range(107)],
}

# Save the dictionary to a JSON file
file_path = '/home/jess/stat_refs_dict.json'
with open(file_path, 'w') as f:
    json.dump(stat_refs_dict, f, indent=4)
