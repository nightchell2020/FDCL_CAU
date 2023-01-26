from tqdm import tqdm

class Dataset(object):
    def __init__(self, name, dataset_root):
        self.name = name
        self.dataset_root = dataset_root
        self.test_signals = None

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.test_signals[idx]
        elif isinstance(idx, int):
            return self.test_signals[sorted(list(self.test_signals.keys()))[idx]]

    def __len__(self):
        return len(self.test_signals)

    def __iter__(self):
        keys = sorted(list(self.test_signals.keys()))
        for key in keys:
            yield self.test_signals[key]

    def set_classificator(self, path, classificator_names):
        """
        Args:
            path: path to tracker results,
            classificator_names: list of classificator name
        """
        self.classificator_path = path
        self.classificator_names = classificator_names
        # for video in tqdm(self.videos.values(), 
        #         desc='loading tacker result', ncols=100):
        #     video.load_tracker(path, tracker_names)
