import torch

def masked_mean(x, mask, dim=None, keepdim=False, return_percentage=False):
    x = x.masked_fill(mask==0, 0)
    mask_sum = mask.sum(dim=dim, keepdim=keepdim)
    mask_sum = mask_sum.masked_fill(mask_sum==0, 1)
    x_mean = x.sum(dim=dim, keepdim=keepdim) / mask_sum

    if return_percentage:
        return x_mean, mask.type(x_mean.dtype).mean(dim=dim)
    return x_mean

def masked_std(x, mask, dim=None, keepdim=False, return_percentage=False, eps=1e-4):
    x = x.masked_fill(mask==0, 0)
    mask_sum = mask.sum(dim=dim, keepdim=True)
    mask_sum = mask_sum.masked_fill(mask_sum==0, 1)
    x_mean = x.sum(dim=dim, keepdim=True) / mask_sum

    x = ((x - x_mean)**2).masked_fill(mask==0, 0)
    x_std = torch.sqrt(x.sum(dim=dim, keepdim=True) / mask_sum + eps)
    if not keepdim:
        x_std = x_std.squeeze(dim)

    if return_percentage:
        return x_std, mask.type(x_mean.dtype).mean(dim=dim)
    return x_std

def torch_nanmean(x, dim=None, keepdim=False, return_percentage=False):
    return masked_mean(x, ~torch.isnan(x), dim=dim, keepdim=keepdim, return_percentage=return_percentage)

def torch_nanstd(x, dim=None, keepdim=False, return_percentage=False, eps=1e-5):
    return masked_std(x, ~torch.isnan(x), dim=dim, keepdim=keepdim, return_percentage=return_percentage, eps=eps)

def clip_outliers(data, dim=0, mask=None, n_sigma=4):
    if mask is None:
        mask = ~torch.isnan(data)
    data_mean = masked_mean(data, mask, dim=dim, keepdim=True)
    data_std = masked_std(data, mask, dim=dim, keepdim=True)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    data = torch.max(-torch.log(1+torch.abs(data)) + lower, data)
    data = torch.min(torch.log(1+torch.abs(data)) + upper, data)
    return data

def normalize_data(data, dim=0, mask=None, mean=None, std=None, transform=True):
    if mask is None:
        mask = ~torch.isnan(data)
    if mean is None:
        mean = masked_mean(data, mask, dim=dim, keepdim=True)
    if std is None:
        std = masked_std(data, mask, dim=dim, keepdim=True)

    if transform:
        return (data - mean) / std
    return mean, std

def process_tensor(data, dim=0, mask=None, mean=None, std=None, transform=True):
    data = clip_outliers(data, dim=dim, mask=mask)
    data = normalize_data(data, dim=dim, mask=mask, mean=mean, std=std, transform=transform)

    return data

def process_data(data, classification=False):
    x_train, y_train, x_test, y_test = data
    """
    x_train: (train_size, feature_size)
    y_train: (train_size,)
    x_test: (test_size, feature_size)
    y_test: (test_size,)
    """
    n_test = x_test.shape[0]

    xs = torch.cat((x_train, x_test), dim=0)
    ys = torch.cat((y_train, y_test), dim=0)
    xs = process_tensor(xs)
    if not classification:
        ys = process_tensor(ys)

    return (
        xs[:-n_test, :],
        ys[:-n_test],
        xs[-n_test:, :],
        ys[-n_test:]
    )


class NumericEncoder:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean, self.std = process_tensor(data, transform=False)
        return self

    def transform(self, data):
        return process_tensor(data, mean=self.mean, std=self.std)

    def fit_transform(self, data, dim=0):
        self.fit(data, dim=dim)
        return self.transform(data)

    def inverse_transform(self, data):
        data = data * self.std + self.mean
        return data


class CategoricalEncoder:
    def __init__(self):
        self.categories = None
        self.category_map = None

    def fit(self, data):
        self.categories = torch.unique(data)
        categories = self.categories.tolist()
        self.category_map = dict(zip(categories, range(len(categories))))
        return self

    def transform(self, data):
        return torch.tensor([self.category_map[cat] for cat in data.tolist()], dtype=data.dtype)

    def fit_transform(self, data):
        self.categories, data = torch.unique(data, return_inverse=True)
        return data

    def inverse_transform(self, data):
        return torch.take(self.categories, data)
