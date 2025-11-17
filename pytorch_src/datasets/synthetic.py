import torch
import pathlib
from tqdm import trange
import numpy as np
from torch_geometric.data import Dataset, Data
from base.dataloader import BaseDataLoader


class SyntheticDataset(Dataset):
    """
    A synthetic dataset generating synthetic graphs according to NPS calorimeter. Clusters consist of a seed and its neighbors in 3x3 grid. Node features mimic FADC time samples.
    """

    def __init__(
        self,
        root: pathlib.Path,
        num_events: int = 100,
        ncols: int = 30,
        nrows: int = 36,
        feature_dim: int = 110,
    ):
        # define before invoking base constructor for proper file naming
        self.num_events = num_events
        self.ncols = ncols
        self.nrows = nrows
        self.nblocks = ncols * nrows
        self.feature_dim = feature_dim

        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)

    @property
    def nrows_(self):
        return self.nrows

    @property
    def ncols_(self):
        return self.ncols

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{i}.pt" for i in range(self.num_events)]

    def download(self):
        pass

    def process(self):

        n_seeds = self.nblocks // 100

        for idx in trange(self.num_events, desc="Generating synthetic graphs"):

            out_pth = pathlib.Path(self.processed_dir) / f"{idx}.pt"
            pos = self._build_positions()  # (N, 2)

            # Baseline noise
            x = torch.normal(0.0, 1.0, size=(self.nblocks, self.feature_dim))

            # Node labels (signal = 1, background = 0)
            node_y = torch.zeros(self.nblocks, dtype=torch.long)

            # Edge list
            edge_list = []

            # Random seeds
            seed_rows = np.random.randint(0, self.nrows, size=n_seeds)
            seed_cols = np.random.randint(0, self.ncols, size=n_seeds)

            used_blocks = set()
            seed_signals = self._generate_seed_signals(n_seeds)

            # cluster sizes (1–8 neighbors)
            n_members = np.random.randint(1, 9, size=n_seeds)

            for i in range(n_seeds):
                seed_idx = seed_rows[i] * self.ncols + seed_cols[i]

                if seed_idx in used_blocks:
                    continue

                used_blocks.add(seed_idx)
                x[seed_idx] += seed_signals[i]
                node_y[seed_idx] = 1

                # Pick neighbors
                neighbors = self._sample_neighbors(
                    seed_rows[i], seed_cols[i], n_members[i]
                )

                # Apply signals w/ attenuation
                for nb in neighbors:
                    if nb not in used_blocks:
                        used_blocks.add(nb)
                        x[nb] += seed_signals[i] * np.random.uniform(0.3, 0.7)
                        node_y[nb] = 1

                cluster = [seed_idx] + neighbors
                self._connect_cluster_fully(edge_list, cluster)

            # Convert edge index
            if len(edge_list) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_list, dtype=torch.long)
                edge_index = edge_index.unique(dim=0)  # remove duplicates
                edge_index = edge_index.t().contiguous()

            graph = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                y=node_y,  # classification: signal vs noise
                edge_attr=None,
            )

            torch.save(graph, out_pth)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _build_positions(self):
        """
        Returns array of (row, col) for each block.
        """
        pos = torch.zeros((self.nblocks, 2), dtype=torch.float32)
        for r in range(self.nrows):
            for c in range(self.ncols):
                idx = r * self.ncols + c
                pos[idx] = torch.tensor([float(r), float(c)])
        return pos

    def _generate_seed_signals(self, n_seeds):
        """
        Generate Gaussian-like FADC pulses.
        """
        signals = []
        for _ in range(n_seeds):
            amp = np.random.uniform(30, 80)
            peak = np.random.randint(20, self.feature_dim - 20)
            sigma = np.random.uniform(4, 10)

            t = np.arange(self.feature_dim)
            pulse = amp * np.exp(-0.5 * ((t - peak) / sigma) ** 2)
            pulse += np.random.normal(0, 0.3, size=self.feature_dim)

            signals.append(torch.tensor(pulse, dtype=torch.float32))

        return torch.stack(signals)

    def _sample_neighbors(self, seed_r, seed_c, n_nb):
        """
        Pick neighbors around the seed within 1-cell radius.
        """
        neighbors = []
        tries = 0

        while len(neighbors) < n_nb and tries < 20:
            tries += 1
            r = np.clip(seed_r + np.random.randint(-1, 2), 0, self.nrows - 1)
            c = np.clip(seed_c + np.random.randint(-1, 2), 0, self.ncols - 1)

            idx = r * self.ncols + c

            if r == seed_r and c == seed_c:
                continue

            if idx not in neighbors:
                neighbors.append(idx)

        return neighbors

    def _connect_cluster_fully(self, edge_list, cluster):
        """
        Add all directed edges inside a cluster.
        """
        if len(cluster) < 2:
            return  # nothing to connect

        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a, b = cluster[i], cluster[j]
                edge_list.append([a, b])
                edge_list.append([b, a])

    def len(self):
        return self.num_events

    def get(self, idx):
        return torch.load(
            pathlib.Path(self.processed_dir) / f"{idx}.pt", weights_only=False
        )


class SyntheticDataLoader(BaseDataLoader):
    def __init__(
        self,
        root: str | pathlib.Path,
        shuffle: bool = True,
        batch_size: int = 32,
        validation_split: float = 0.0,
        num_workers: int = 1,
    ):
        dataset = SyntheticDataset(root=str(root))
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

    @property
    def num_features_(self):
        return self.dataset[0].num_features

    @property
    def num_classes_(self):
        return 2
