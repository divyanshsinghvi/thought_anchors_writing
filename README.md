# MATS pair project-1 : Extending Thought Anchors

## Generate Samples

This project generates writing samples using the rollouts library with different system prompts.

### Commands

**For non-fiction (sync mode):**
```bash
python generate_samples.py --type non-fiction
```

**For non-fiction (async mode - faster):**
```bash
python generate_samples.py --type non-fiction --async
```

**For fiction (when you create fictional_prompts.yaml):**
```bash
python generate_samples.py --type fiction --async
```

### Configuration

Edit `config.py` to customize:
- Output directories
- Cache path
- Model name
- Input YAML files

### Output Structure

Generated samples are saved in JSON format to:
- `/mnt/d/code/open_source/mats/thought_anchors_writing/data/non_fiction/sys_prompt_1/`
- `/mnt/d/code/open_source/mats/thought_anchors_writing/data/non_fiction/sys_prompt_2/`
- `/mnt/d/code/open_source/mats/thought_anchors_writing/data/fiction/sys_prompt_1/`
- `/mnt/d/code/open_source/mats/thought_anchors_writing/data/fiction/sys_prompt_2/`

Each JSON file contains:
- `prompt`: The combined system and user prompt
- `response`: The generated response
- `model`: Model used for generation
- `system_prompt`: Which system prompt was used
- `sample_id`: Sample number
- `prompt_name`: Short name extracted from prompt
