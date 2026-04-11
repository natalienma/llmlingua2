from datasets import load_dataset
import json

ds = load_dataset("microsoft/MeetingBank-LLMCompressed")

with open("meetingbank.json", "w") as f:
    json.dump(ds['train'][0], f, indent=2)


