import torch.utils.data as data


def get_batch_loader(features, labels, batch_size, shuffle=True):
    class MyDataset(data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __getitem__(self, index):  # tensor
            row_data, target = self.features[index], self.labels[index]
            return row_data, target

        def __len__(self):
            return len(self.features)

    batch_loader = data.DataLoader(MyDataset(features, labels), batch_size=batch_size, shuffle=shuffle)
    return batch_loader

