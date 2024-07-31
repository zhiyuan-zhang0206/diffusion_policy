from pathlib import Path
import pickle

ROBOMIMIC_TASK_TO_LANGUAGE = {
    'lift': 'pick up the cube',
    'can': 'pick up the can and place it to the required place',
    'square': 'pick up the handle and position the square frame over the standing cuboid',
}

def parse_task_from_dataset_path(path: str):
    task = path.split('/')[-3]
    assert task in ROBOMIMIC_TASK_TO_LANGUAGE, f'Task {task} not defined'
    return task

def get_language_embedding(dataset_path: str):
    """
    dataset_path: the path to the dataset.
    Automatically find the language embedding file near the dataset path.
    """
    language_embedding_path = Path(dataset_path).parent.parent / 'language_embedding.pkl'
    if not language_embedding_path.exists():
        raise FileNotFoundError(f'Language embedding file not found at {language_embedding_path}')
    with language_embedding_path.open('rb') as f:
        embedding = pickle.load(f)
    return embedding

def get_language_description(dataset_path:str):
    task = parse_task_from_dataset_path(dataset_path)
    return ROBOMIMIC_TASK_TO_LANGUAGE[task]

def main():
    
    import os
    os.environ["HF_HOME"] = "/home/zzy/robot/data/.huggingface"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    import sys
    sys.path.append(Path(__file__).parent.parent.parent.as_posix())
    from diffusion_policy.model.language.DistilBERT_utils import DistilBERTWrapper
    
    robomimic_path = Path('/home/zzy/robot/data/diffusion_policy_data/data/robomimic/datasets')
    model = DistilBERTWrapper()
    for task in ROBOMIMIC_TASK_TO_LANGUAGE.keys():
        embedding = model(task)[0]
        save_path = robomimic_path / task / 'language_embedding.pkl'
        with save_path.open('wb') as f:
            pickle.dump(embedding.detach().cpu().numpy(), f)
        print(embedding.shape)

if __name__ == '__main__':
    main()