import json
# Mapping from simplified sequential IDs [1, 2, ...] to original fineweb-id values
data_raw_path = "/* path to where the raw data provided is saved*/" 
mapping_output_path = "fineweb_id_mapping.json"

id_mapping = {}  # {converted_id: fineweb-id}

with open(data_raw_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        id_mapping[obj["id"]] = obj["fineweb-id"]

with open(mapping_output_path, "w") as f:
    json.dump(id_mapping, f, indent=2)

print(f"Saved mapping of {len(id_mapping)} ids to {mapping_output_path}")
