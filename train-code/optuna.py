import torch




def objective(trial):
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
