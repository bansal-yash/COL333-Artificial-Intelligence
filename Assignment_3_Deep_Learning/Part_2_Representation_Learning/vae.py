import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import pickle
import sys

# python3 vae.py mnist_1_4_8_train.npz mnist_1_4_8_val_recon.npz train vae.pth gmm.pkl
# python3 vae.py mnist_1_4_8_train.npz test_reconstruction vae.pth
# python3 vae.py mnist_1_4_8_train.npz test_classifier vae.pth gmm.pkl

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.cuda.empty_cache()


class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        image = image.unsqueeze(0)
        return image


class ImageDatasetWithLabels(Dataset):
    def __init__(self, images, labels):
        self.images = images / 255.0
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class col333_vae(nn.Module):
    def __init__(self):
        super(col333_vae, self).__init__()

        self.common_fc = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.BatchNorm1d(400),
            nn.Tanh(),
            nn.Linear(400, 196),
            nn.BatchNorm1d(196),
            nn.Tanh(),
            nn.Linear(196, 48),
            nn.BatchNorm1d(48),
            nn.Tanh(),
        )
        self.mean_fc = nn.Sequential(
            nn.Linear(48, 16), nn.BatchNorm1d(16), nn.Tanh(), nn.Linear(16, 2)
        )
        self.log_var_fc = nn.Sequential(
            nn.Linear(48, 16), nn.BatchNorm1d(16), nn.Tanh(), nn.Linear(16, 2)
        )

        self.decoder_fcs = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Linear(16, 48),
            nn.BatchNorm1d(48),
            nn.Tanh(),
            nn.Linear(48, 196),
            nn.BatchNorm1d(196),
            nn.Tanh(),
            nn.Linear(196, 400),
            nn.BatchNorm1d(400),
            nn.Tanh(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x):
        out = self.common_fc(torch.flatten(x, start_dim=1))
        mean = self.mean_fc(out)
        log_var = self.log_var_fc(out)
        return mean, log_var

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        z = torch.randn_like(std)
        z = z * std + mean
        return z

    def decode(self, z):
        out = self.decoder_fcs(z)
        out = out.reshape((z.size(0), 1, 28, 28))
        return out

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.sample(mean, log_var)
        out = self.decode(z)
        return mean, log_var, out


def gmm(model: col333_vae, train_loader, classifier_loader):
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for images in train_loader:
            images = images.to(device)
            mean, _, _ = model(images)
            latent_vectors.append(mean)

    train_latent_vectors = torch.cat(latent_vectors).cpu().numpy()

    latent_vectors, labels = [], []
    with torch.no_grad():
        for images, label in classifier_loader:
            images = images.to(device)
            mean, _, _ = model(images)
            latent_vectors.append(mean)
            labels.append(label)
    classifier_latent_vectors = torch.cat(latent_vectors).cpu().numpy()
    classifier_labels = torch.cat([label.view(-1) for label in labels]).cpu().numpy()

    cluster_labels = np.unique(classifier_labels)
    num_clusters = len(cluster_labels)

    mean_dict = {
        label: classifier_latent_vectors[classifier_labels == label].mean(axis=0)
        for label in cluster_labels
    }
    mean_vectors = np.stack([mean_dict[label] for label in cluster_labels])
    covariances = np.stack(
        [0.1 * np.eye(train_latent_vectors.shape[1]) for _ in range(num_clusters)]
    )
    mixing_coeffs = np.ones(num_clusters) / num_clusters

    tol = 1e-15

    for iteration in range(1000):
        responsibilities = np.zeros((train_latent_vectors.shape[0], num_clusters))

        for i in range(num_clusters):
            mean = mean_vectors[i]
            covariance = covariances[i]
            coeff = mixing_coeffs[i]

            diff = train_latent_vectors - mean
            try:
                exponent = -0.5 * np.sum(
                    diff @ np.linalg.inv(covariance) * diff, axis=1
                )
                normalization = np.sqrt(np.linalg.det(2 * np.pi * covariance))
            except np.linalg.LinAlgError:
                continue

            responsibilities[:, i] = coeff * np.exp(exponent) / normalization

        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        total_responsibility = np.sum(responsibilities, axis=0)
        mean_vectors_new = (
            np.dot(responsibilities.T, train_latent_vectors)
            / total_responsibility[:, np.newaxis]
        )

        covariances_new = np.zeros_like(covariances)
        for i in range(num_clusters):
            diff = train_latent_vectors - mean_vectors_new[i]
            covariances_new[i] = (
                (responsibilities[:, i] * diff.T) @ diff / total_responsibility[i]
            )
            covariances_new[i] += 1e-6 * np.eye(covariances_new[i].shape[0])

        mixing_coeffs_new = total_responsibility / train_latent_vectors.shape[0]

        mean_change = np.linalg.norm(mean_vectors_new - mean_vectors) / np.linalg.norm(
            mean_vectors
        )
        cov_change = np.linalg.norm(covariances_new - covariances) / np.linalg.norm(
            covariances
        )
        coeff_change = np.linalg.norm(
            mixing_coeffs_new - mixing_coeffs
        ) / np.linalg.norm(mixing_coeffs)

        if mean_change < tol and cov_change < tol and coeff_change < tol:
            print(f"Converged after {iteration + 1} iterations")
            break

        mean_vectors, covariances, mixing_coeffs = (
            mean_vectors_new,
            covariances_new,
            mixing_coeffs_new,
        )

    return mean_vectors, covariances, mixing_coeffs, cluster_labels


def make_test_predictions(
    model, test_loader, mean_vectors, covariances, mixing_coeffs, cluster_labels
):
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            mean, _, _ = model(images)
            latent_vectors.append(mean)

    data_latent_vectors = torch.cat(latent_vectors).cpu().numpy()

    num_clusters = len(cluster_labels)
    responsibilities = np.zeros((data_latent_vectors.shape[0], num_clusters))

    for i in range(num_clusters):
        mean = mean_vectors[i]
        covariance = covariances[i]
        coeff = mixing_coeffs[i]

        diff = data_latent_vectors - mean
        exponent = -0.5 * np.sum(diff @ np.linalg.inv(covariance) * diff, axis=1)
        normalization = np.sqrt(np.linalg.det(2 * np.pi * covariance))
        responsibilities[:, i] = coeff * np.exp(exponent) / normalization

    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    cluster_assignments = np.argmax(responsibilities, axis=1)
    cluster_assignments = np.array([cluster_labels[i] for i in cluster_assignments])

    return cluster_assignments


def loss_function(recon_x, x, mu, logvar):
    bce = nn.functional.binary_cross_entropy(
        recon_x.view(-1, 784), x.view(-1, 784), reduction="sum"
    )
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld


def train_vae(model: col333_vae, train_loader, optimizer, num_epochs):
    num_images_train = len(train_loader.dataset)
    for epoch_idx in range(num_epochs):
        model.train()

        running_train_loss = 0
        running_recon_loss = 0
        running_kl_loss = 0
        running_train_mse = 0

        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            mean, log_var, out = model(images)

            loss, bce, kld = loss_function(out, images, mean, log_var)

            running_train_loss += loss.item()
            running_recon_loss += bce.item()
            running_kl_loss += kld.item()

            mse_loss = torch.nn.functional.mse_loss(out, images, reduction="sum")
            running_train_mse += mse_loss.item()

            loss.backward()
            optimizer.step()

        running_train_loss /= num_images_train
        running_recon_loss /= num_images_train
        running_kl_loss /= num_images_train
        running_train_mse /= num_images_train

        print(f"\nEpoch: {epoch_idx + 1}/{num_epochs}")
        print(
            f"Total Loss: {running_train_loss:4f}, recon_loss: {running_recon_loss:4f}, kl_loss: {running_kl_loss:4f}"
        )
        print(f"Train MSE: {running_train_mse}")


def test_recon(model, test_loader):
    model.eval()
    recon_images = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            _, _, out = model(images)
            recon_images.append(out)

    recon_images = torch.cat(recon_images).squeeze(1).cpu().numpy()
    return recon_images


if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None

    if len(sys.argv) == 4:  ### Running code for vae reconstruction.
        path_to_test_dataset_recon = arg1
        test_reconstruction = arg2
        vaePath = arg3

        model = col333_vae()
        model = model.to(device)

        model.load_state_dict(torch.load(vaePath))

        test_data = np.load(path_to_test_dataset_recon)

        test_dataset = ImageDataset(test_data["data"])
        test_loader = DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=4
        )

        recon_images = test_recon(model, test_loader)
        np.savez("vae_reconstructed.npz", data=recon_images)

    elif len(sys.argv) == 5:  ###Running code for class prediction during testing
        path_to_test_dataset = arg1
        test_classifier = arg2
        vaePath = arg3
        gmmPath = arg4

        with open(gmmPath, "rb") as f:
            gmm_data = pickle.load(f)

        mean_vectors, covariances, mixing_coeffs, cluster_labels = (
            gmm_data["mean_vectors"],
            gmm_data["covariances"],
            gmm_data["mixing_coeffs"],
            gmm_data["cluster_labels"],
        )

        model = col333_vae()
        model = model.to(device)

        model.load_state_dict(torch.load(vaePath))

        test_data = np.load(path_to_test_dataset)
        test_dataset = ImageDataset(test_data["data"])
        test_loader = DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=4
        )

        predictions = make_test_predictions(
            model, test_loader, mean_vectors, covariances, mixing_coeffs, cluster_labels
        )

        with open("vae.csv", "w") as f:
            f.write("Predicted_Label\n")
            for label in predictions:
                f.write(f"{label}\n")

    else:  ### Running code for training. save the model in the same directory with name "vae.pth"
        path_to_train_dataset = arg1
        path_to_val_dataset = arg2
        trainStatus = arg3
        vaePath = arg4
        gmmPath = arg5

        train_data = np.load(path_to_train_dataset)
        classifier_data = np.load(path_to_val_dataset)

        train_dataset = ImageDataset(train_data["data"])
        train_loader = DataLoader(
            train_dataset, batch_size=512, shuffle=True, num_workers=4
        )

        classifier_dataset = ImageDatasetWithLabels(
            classifier_data["data"], classifier_data["labels"]
        )
        classifier_loader = DataLoader(
            classifier_dataset, batch_size=100, shuffle=False, num_workers=4
        )

        model = col333_vae()
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 83

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {num_params}")

        train_vae(model, train_loader, optimizer, num_epochs)

        mean_vectors, covariances, mixing_coeffs, cluster_labels = gmm(
            model, train_loader, classifier_loader
        )

        torch.save(model.state_dict(), vaePath)

        gmm_data = {
            "mean_vectors": mean_vectors,
            "covariances": covariances,
            "mixing_coeffs": mixing_coeffs,
            "cluster_labels": cluster_labels,
        }

        with open(gmmPath, "wb") as f:
            pickle.dump(gmm_data, f)
