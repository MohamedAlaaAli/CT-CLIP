from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer
from accelerate import Accelerator, DistributedDataParallelKwargs
import tqdm
import pandas as pd
from ct_clip import CTCLIP
from eval import evaluate_internal
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_nii_volume(nii_path, target_size=(480, 480, 480), normalize=True):
    """
    Loads a NIfTI volume and resizes it to (D, H, W).
    If target_size[2] is None, keeps the original depth.
    
    Args:
        nii_path (str | Path): Path to .nii or .nii.gz file.
        target_size (tuple): (H, W, D) target size; set D=None to keep original.
        normalize (bool): Normalize voxel intensities to [0, 1].
    
    Returns:
        np.ndarray: 3D volume array (D, H, W) as float32
    """
    nii = nib.load(str(nii_path))
    volume = nii.get_fdata()

    # Ensure shape is (D, H, W)
    if volume.ndim == 2:
        volume = volume[None, :, :]
    elif volume.ndim > 3:
        volume = np.squeeze(volume)

    D, H, W = volume.shape
    target_H, target_W, target_D = target_size

    # ---- Normalize ----
    if normalize:
        vmin, vmax = np.percentile(volume, (1, 99))
        volume = np.clip((volume - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # ---- Compute zoom factors ----
    zoom_D = 1.0 if target_D is None else target_D / D
    zoom_factors = (zoom_D, target_H / H, target_W / W)

    # ---- Resize ----
    volume_resized = zoom(volume, zoom_factors, order=1)
    return volume_resized.astype(np.float32)




def apply_softmax(array):
    softmax = torch.nn.Softmax(dim=0)
    return softmax(array)


def noop(*args, **kwargs):
    pass


class CTClipInference(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        data_folder: str,
        results_folder: str = "./results",
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True
        )

        self.data_folder = Path(data_folder)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.register_buffer("steps", torch.Tensor([0]))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.CTClip.to(self.device)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # def infer(self, log_fn=noop):
    #     device = self.device
    #     steps = int(self.steps.item())
    #     logs = {}

    #     nii_files = sorted(list(self.data_folder.glob("*.nii*")))
    #     if not nii_files:
    #         raise FileNotFoundError(f"No .nii or .nii.gz files found in {self.data_folder}")

    #     pathologies = [
    #         "Medical material", "Arterial wall calcification", "Cardiomegaly",
    #         "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    #         "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    #         "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    #         "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    #         "Interlobular septal thickening"
    #     ]

    #     with torch.no_grad():
    #         model = self.CTClip
    #         model.eval()

    #         predictedall = []
    #         accession_names = []

    #         for nii_path in tqdm.tqdm(nii_files, desc="Running inference on NIfTI volumes"):
    #             # Load volume
    #             volume = load_nii_volume(nii_path)
    #             volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    #             predictedlabels = []

    #             # Inference for each pathology
    #             for pathology in pathologies:
    #                 text = [f"{pathology} is present.", f"{pathology} is not present."]
    #                 text_tokens = self.tokenizer(
    #                     text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
    #                 ).to(device)

    #                 output = model(text_tokens, volume_tensor, device=device)
    #                 output = apply_softmax(output)
    #                 predictedlabels.append(output.detach().cpu().numpy()[0])

    #             predictedall.append(predictedlabels)
    #             accession_names.append(nii_path.stem)

    #         predictedall = np.array(predictedall)
    #         plotdir = self.results_folder
    #         np.savez(plotdir / "predicted_weights.npz", data=predictedall)

    #         with open(plotdir / "accessions.txt", "w") as f:
    #             for name in accession_names:
    #                 f.write(name + "\n")

    #     self.steps += 1
    #     log_fn(logs)
    #     print("Inference complete ✅")

    def infer(self, log_fn=noop):
        device = self.device
        steps = int(self.steps.item())
        logs = {}

        nii_files = sorted(list(self.data_folder.glob("*.nii*")))
        if not nii_files:
            raise FileNotFoundError(f"No .nii or .nii.gz files found in {self.data_folder}")

        pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 
                        'Pericardial effusion','Coronary artery wall calcification', 
                        'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 
                        'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 
                        'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 
                        'Consolidation', 'Bronchiectasis','Interlobular septal thickening', "Pulmonary embolism"]


        results = []

        with torch.no_grad():
            model = self.CTClip
            model.eval()

            for nii_path in tqdm.tqdm(nii_files, desc="Running inference on NIfTI volumes"):
                volume = load_nii_volume(nii_path)
                volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                row = {"File": nii_path.stem}
                probs_list = []

                for pathology in pathologies:
                    text = [f"{pathology} is present.", f"{pathology} is not present."]
                    text_tokens = self.tokenizer(
                        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
                    ).to(device)

                    output = model(text_tokens, volume_tensor, device=device)
                    probs = apply_softmax(output).detach().cpu().numpy()
                    present_prob = float(probs[0]) if probs.ndim > 0 else float(probs)
                    row[pathology] = round(present_prob, 4)
                    probs_list.append(present_prob)

                results.append(row)

                # ---- Print max label ----
                max_idx = np.argmax(probs_list)
                max_pathology = pathologies[max_idx]
                max_prob = probs_list[max_idx]
                print(f"{nii_path.stem}: Max predicted pathology -> {max_pathology} ({max_prob:.4f})")

                # ---- Horizontal bar plot with top 3 highlighted and annotations ----
                plt.figure(figsize=(10, 8))
                top3_idx = np.argsort(probs_list)[-3:]
                colors = ["skyblue"] * len(pathologies)
                for idx in top3_idx:
                    colors[idx] = "orange"

                bars = plt.barh(pathologies, probs_list, color=colors)
                plt.xlabel("Probability")
                plt.title(f"{nii_path.stem} - Predicted Pathologies")
                plt.xlim(0, 1)
                plt.gca().invert_yaxis()  # highest probability on top

                # Annotate values on bars
                for bar, prob in zip(bars, probs_list):
                    plt.text(prob + 0.01, bar.get_y() + bar.get_height()/2, f"{prob:.2f}", va='center')

                plt.tight_layout()
                plot_path = self.results_folder / f"{nii_path.stem}_barplot.png"
                plt.savefig(plot_path)
                plt.close()

        # ---- Save CSV ----
        results_df = pd.DataFrame(results)
        csv_path = self.results_folder / "predictions.csv"
        results_df.to_csv(csv_path, index=False)

        self.steps += 1
        log_fn(logs)
        print(f"Inference complete ✅ Results saved to: {csv_path}")


