#export OMP_NUM_THREADS=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

from tools import load_file, save_file, get_fsct_path
from model import Net
from train_datasets import TrainingDataset, ValidationDataset
from fsct_exceptions import NoDataFound
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import glob
import random
import threading
import os
import shutil
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# os.environ["OMP_NUM_THREADS"]=1
from torch.utils.data import BatchSampler
from torch_geometric.data import Batch

def collate_fn(batch):
    return Batch.from_data_list(batch)
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
dist.init_process_group(
    backend="nccl",       # recommended for GPUs
    world_size=world_size,
    rank=rank
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    state = {
        'base_model': model.module.state_dict() ,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc,
    }
    torch.save(state, path)
    print(f"Saved checkpoint at {path}")





from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
import torch
import numpy as np


# class DynamicBatchSampler(BatchSampler):
#     """BatchSampler that adjusts batch size dynamically based on VRAM."""
#
#     def __init__(self, dataset, sampler, max_batch_vram_mb=2.6, drop_last=False):
#         self.dataset = dataset
#         self.sampler = sampler
#         self.max_batch_vram_mb = max_batch_vram_mb
#         self.drop_last = drop_last
#
#     def _get_sample_vram_mb(self, idx):
#         sample = self.dataset[idx]
#         total_bytes = sum(attr.element_size() * attr.nelement()
#                           for attr in sample.__dict__.values() if torch.is_tensor(attr))
#         return total_bytes / (1024 ** 2)
#
#     def __iter__(self):
#         batch = []
#         current_vram = 0.0
#         for idx in self.sampler:
#             vram = self._get_sample_vram_mb(idx)
#             if current_vram + vram > self.max_batch_vram_mb:
#                 if len(batch) > 0:
#                     yield batch
#                 batch = [idx]
#                 current_vram = vram
#             else:
#                 batch.append(idx)
#                 current_vram += vram
#         if len(batch) > 0 and not self.drop_last:
#             yield batch
#
#     def __len__(self):
#         total_vram = sum(self._get_sample_vram_mb(idx) for idx in self.sampler)
#         return int(np.ceil(total_vram / self.max_batch_vram_mb))

# from torch.utils.data import Sampler
# from torch.utils.data import BatchSampler
# from torch_geometric.data import Batch

# class DynamicBatchSampler(BatchSampler):
#     """
#     BatchSampler that adjusts batch size dynamically based on estimated VRAM.
#     Supports DistributedSampler.
#     """
#
#     def __init__(self, dataset, sampler, max_batch_vram_mb=2000, drop_last=False):
#         """
#         Args:
#             dataset: PyG dataset
#             sampler: Sampler (can be DistributedSampler)
#             max_batch_vram_mb: approximate max VRAM per batch in MB (after PyG concatenation)
#             drop_last: whether to drop the last incomplete batch
#         """
#         self.dataset = dataset
#         self.sampler = sampler
#         self.max_batch_vram_mb = max_batch_vram_mb
#         self.drop_last = drop_last
#
#     def _get_sample_vram_mb(self, idx):
#         sample = self.dataset[idx]
#         total_bytes = sum(
#             attr.element_size() * attr.nelement()
#             for attr in sample.__dict__.values()
#             if torch.is_tensor(attr)
#         )
#         return total_bytes / (1024 ** 2)
#
#     def __iter__(self):
#         batch = []
#         current_vram = 0.0
#         for idx in iter(self.sampler):
#             sample_vram = self._get_sample_vram_mb(idx)
#             # Safety margin: multiply by ~10 to approximate PyG batching overhead
#             estimated_batch_vram = current_vram + sample_vram * 10
#             if estimated_batch_vram > self.max_batch_vram_mb:
#                 if len(batch) > 0:
#                     yield batch
#                 batch = [idx]
#                 current_vram = sample_vram
#             else:
#                 batch.append(idx)
#                 current_vram += sample_vram
#         if len(batch) > 0 and not self.drop_last:
#             yield batch
#
#     def __len__(self):
#         total_vram = sum(self._get_sample_vram_mb(idx) for idx in self.sampler)
#         return int(np.ceil(total_vram * 10 / self.max_batch_vram_mb))


class DynamicBatchSampler(BatchSampler):
    """
    BatchSampler that dynamically builds batches based on estimated sample size in MB.
    """

    def __init__(self, dataset, sampler, max_batch_mb=2.6, drop_last=False):
        self.dataset = dataset
        self.sampler = sampler
        self.max_batch_mb = max_batch_mb
        self.drop_last = drop_last

        # Precompute sample sizes in MB
        self.sample_sizes = [self._estimate_sample_size(dataset[i]) for i in range(len(dataset))]

    def _estimate_sample_size(self, data):
        file_path = getattr(data, "file_path", None)
        if file_path is None:
            raise ValueError("Data object must have 'file_path' attribute")
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 ** 2)

    def __iter__(self):
        batch = []
        batch_size = 0.0
        for idx in self.sampler:
            sample_size = self.sample_sizes[idx]

            if batch_size + sample_size > self.max_batch_mb:

                if batch:
                    yield batch
                batch = [idx]
                batch_size = sample_size

            else:
                batch.append(idx)
                batch_size += sample_size

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total_size = sum(self.sample_sizes[idx] for idx in self.sampler)
        return int((total_size + self.max_batch_mb - 1) // self.max_batch_mb)

class TrainModel:
    def __init__(self, parameters):
        self.parameters = parameters

        if self.parameters["num_cpu_cores_preprocessing"] == 0:
            print("Using default number of CPU cores (all of them).")
            self.parameters["num_cpu_cores_preprocessing"] = os.cpu_count()
        print("Processing using ", self.parameters["num_cpu_cores_preprocessing"], "/", os.cpu_count(), " CPU cores.")

        if self.parameters["preprocess_train_datasets"]:
            self.preprocessing_setup("train")

        if self.parameters["preprocess_validation_datasets"]:
            self.preprocessing_setup("validation")

        self.device = parameters["device"]

    def check_and_fix_data_directory_structure(self, data_sub_directory):
        """
        Creates the data directory and required subdirectories in
        the FSCT directory if they do not already exist.
        """
        fsct_dir = get_fsct_path()
        dir_list = [
            os.path.join(fsct_dir, "data"),
            os.path.join(fsct_dir, "data", data_sub_directory),
            os.path.join(fsct_dir, "data", data_sub_directory, "sample_dir"),
        ]

        for directory in dir_list:
            if not os.path.isdir(directory):
                os.makedirs(directory)
                print(directory, "directory created.")

            elif "sample_dir" in directory and self.parameters["clean_sample_directories"]:
                shutil.rmtree(directory, ignore_errors=True)
                os.makedirs(directory)
                print(directory, "directory created.")

            else:
                print(directory, "directory found.")

    def preprocessing_setup(self, data_subdirectory):
        self.check_and_fix_data_directory_structure(data_subdirectory)
        point_cloud_list = glob.glob(get_fsct_path("data") + "/" + data_subdirectory + "/*.las")
        if len(point_cloud_list) > 0:
            print("Preprocessing train_dataset point clouds...")
            for point_cloud_file in point_cloud_list:
                print(point_cloud_file)
                point_cloud, headers = load_file(point_cloud_file, headers_of_interest=["x", "y", "z", "classification"])
                self.preprocess_point_cloud(
                    point_cloud, get_fsct_path("data") + "/" + data_subdirectory + "/sample_dir/"
                )

    @staticmethod
    def threaded_boxes(point_cloud, box_size, min_points_per_box, max_points_per_box, path, id_offset, point_divisions):
        box_size = np.array(box_size)
        box_centre_mins = point_divisions - 0.5 * box_size
        box_centre_maxes = point_divisions + 0.5 * box_size
        i = 0
        pds = len(point_divisions)
        while i < pds:
            box = point_cloud
            box = box[
                np.logical_and(
                    np.logical_and(
                        np.logical_and(box[:, 0] >= box_centre_mins[i, 0], box[:, 0] < box_centre_maxes[i, 0]),
                        np.logical_and(box[:, 1] >= box_centre_mins[i, 1], box[:, 1] < box_centre_maxes[i, 1]),
                    ),
                    np.logical_and(box[:, 2] >= box_centre_mins[i, 2], box[:, 2] < box_centre_maxes[i, 2]),
                )
            ]

            if box.shape[0] > min_points_per_box:
                if box.shape[0] > max_points_per_box:
                    indices = list(range(0, box.shape[0]))
                    random.shuffle(indices)
                    random.shuffle(indices)
                    box = box[indices[:max_points_per_box], :]
                    box = np.asarray(box, dtype="float32")
                np.save(path + str(id_offset + i).zfill(7) + ".npy", box)
            i += 1
        return 1

    def global_shift_to_origin(self, point_cloud):
        point_cloud_mins = np.min(point_cloud[:, :3], axis=0)
        point_cloud_maxes = np.max(point_cloud[:, :3], axis=0)
        point_cloud_ranges = point_cloud_maxes - point_cloud_mins
        point_cloud_centre = point_cloud_mins + 0.5 * point_cloud_ranges

        point_cloud[:, :3] = point_cloud[:, :3] - point_cloud_centre
        return point_cloud, point_cloud_centre

    def preprocess_point_cloud(self, point_cloud, sample_dir):
        def get_box_centre_list(point_cloud_mins, num_boxes_array):
            box_centre_list = []
            for (dimension_min, dimension_num_boxes, box_size_m, box_overlap) in zip(
                point_cloud_mins,
                num_boxes_array,
                self.parameters["sample_box_size_m"],
                self.parameters["sample_box_overlap"],
            ):
                box_centre_list.append(
                    np.linspace(
                        dimension_min,
                        dimension_min + (int(dimension_num_boxes) * box_size_m),
                        int(int(dimension_num_boxes) / (1 - box_overlap)) + 1,
                    )
                )
            return box_centre_list

        print("Pre-processing point cloud...")
        # Global shift the point cloud to avoid loss of precision during segmentation.
        point_cloud, _ = self.global_shift_to_origin(point_cloud)

        point_cloud_mins = np.min(point_cloud[:, :3], axis=0)
        point_cloud_maxes = np.max(point_cloud[:, :3], axis=0)
        point_cloud_ranges = point_cloud_maxes - point_cloud_mins
        point_cloud_centre = point_cloud_mins + 0.5 * point_cloud_ranges

        num_boxes_array = np.ceil(point_cloud_ranges / self.parameters["sample_box_size_m"])
        box_centres = np.vstack(np.meshgrid(*get_box_centre_list(point_cloud_mins, num_boxes_array))).reshape(3, -1).T

        point_divisions = []
        for thread in range(self.parameters["num_cpu_cores_preprocessing"]):
            point_divisions.append([])

        points_to_assign = box_centres

        while points_to_assign.shape[0] > 0:
            for i in range(self.parameters["num_cpu_cores_preprocessing"]):
                point_divisions[i].append(points_to_assign[0, :])
                points_to_assign = points_to_assign[1:]
                if points_to_assign.shape[0] == 0:
                    break
        threads = []
        id_offset = 0
        training_data_list = glob.glob(sample_dir + "*.npy")
        if len(training_data_list) > 0:
            id_offset = np.max([int(os.path.basename(i).split(".")[0]) for i in training_data_list]) + 1

        for thread in range(self.parameters["num_cpu_cores_preprocessing"]):
            for t in range(thread):
                id_offset = id_offset + len(point_divisions[t])
            t = threading.Thread(
                target=self.threaded_boxes,
                args=(
                    point_cloud,
                    self.parameters["sample_box_size_m"],
                    self.parameters["min_points_per_box"],
                    self.parameters["max_points_per_box"],
                    sample_dir,
                    id_offset,
                    point_divisions[thread],
                ),
            )
            threads.append(t)

        for x in threads:
            x.start()

        for x in threads:
            x.join()

    def update_log(self, epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc):
        self.training_history = np.vstack(
            (self.training_history, np.array([[epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc]]))
        )
        try:
            np.savetxt(os.path.join(get_fsct_path("model"), "training_history.csv"), self.training_history)
        except PermissionError:
            print("training_history not saved this epoch, please close training_history.csv to enable saving.")
            try:
                np.savetxt(
                    os.path.join(get_fsct_path("model"), "training_history_permission_error_backup.csv"),
                    self.training_history,
                )
            except PermissionError:
                pass

    def run_training(self):
        if self.parameters["num_cpu_cores_deep_learning"] == 0:
            print("Using default number of CPU cores (all of them).")
            self.parameters["num_cpu_cores_deep_learning"] = os.cpu_count()
        print(
            "Running deep learning using ",
            self.parameters["num_cpu_cores_deep_learning"],
            "/",
            os.cpu_count(),
            " CPU cores.",
        )

        self.training_history = np.zeros((0, 5))
        ###########
        train_dataset = TrainingDataset(
            root_dir=os.path.join(get_fsct_path("data"), "train/sample_dir/"),
            device=self.device,
            min_sample_points=self.parameters["min_points_per_box"],
            max_sample_points=parameters["max_points_per_box"],
        )

        train_sampler = DistributedSampler(train_dataset)
        dynamic_sampler = DynamicBatchSampler(train_dataset, train_sampler, max_batch_mb=2.6)

        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=dynamic_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,  # <- important!
        )

        if len(train_dataset) == 0:
            raise NoDataFound("No training samples found.")



        if self.parameters["perform_validation_during_training"]:
            validation_dataset = ValidationDataset(
                root_dir=os.path.join(get_fsct_path("data"), "validation/sample_dir/"),
                device=self.device,
            )

            if len(validation_dataset) == 0:
                raise NoDataFound("No validation samples found.")

            ######
            self.validation_loader = DataLoader(
                validation_dataset,
                batch_size=parameters["validation_batch_size"],
                shuffle=True,
                num_workers=self.parameters["num_cpu_cores_deep_learning"],
                drop_last=True,
                collate_fn=collate_fn  # <--- add this
            )

        model = Net(num_classes=4).to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

        if self.parameters["load_existing_model"]:
            checkpoint = torch.load(os.path.join(get_fsct_path("model"), self.parameters["model_filename_load"]))
            model.module.load_state_dict(checkpoint['base_model'], strict=False)

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=self.parameters["learning_rate"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=self.parameters["learning_rate"])

        criterion = nn.CrossEntropyLoss()
        val_epoch_loss = 0
        val_epoch_acc = 0

        if self.parameters["load_existing_model"]:
            best_val_acc=torch.load("./model/best_model.pth")["val_acc"]
        else:
            best_val_acc=0

        for epoch in range(self.parameters["num_epochs"]):
            train_sampler.set_epoch(epoch)
            start_time_train = time.time()
            print("=====================================================================")
            print("EPOCH ", epoch)
            # TRAINING
            model.train()
            running_loss = 0.0
            running_acc = 0
            i = 0
            running_point_cloud_vis = np.zeros((0, 5))
            for data in self.train_loader:
                data = data.to(local_rank)
                data.y = torch.unsqueeze(data.y, 0)
                outputs = model(data)
                loss = criterion(outputs, data.y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach().item()
                running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
                # running_point_cloud_vis = np.vstack(
                #     (
                #         running_point_cloud_vis,
                #         np.hstack((data.pos.cpu() + np.array([i * 7, 0, 0]), data.y.cpu().T, preds.cpu().T)),
                #     )
                # )






                print(
                    "Train sample accuracy: ",
                    np.around(running_acc / (i + 1), 4),
                    ", Loss: ",
                    np.around(running_loss / (i + 1), 4),
                )
                #
                # if self.parameters["generate_point_cloud_vis"]:
                #     save_file(
                #         os.path.join(get_fsct_path("data"), "latest_prediction.las"),
                #         running_point_cloud_vis,
                #         headers_of_interest=["x", "y", "z", "label", "prediction"],
                #     )





            total_loss = torch.tensor(running_loss, device=local_rank)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            epoch_loss = total_loss.item() / world_size

            epoch_acc = running_acc / len(self.train_loader)



            self.update_log(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc)
            # print("Train epoch accuracy: ", np.around(epoch_acc, 4), ", Loss: ", np.around(epoch_loss, 4), "\n")

            torch.cuda.empty_cache()
            # VALIDATION
            if local_rank == 0:
                print("Validation")

                if self.parameters["perform_validation_during_training"]:
                    model.eval()
                    running_loss = 0.0
                    running_acc = 0
                    i = 0
                    device = torch.device(f"cuda:{local_rank}")
                    model.to(device)
                    with torch.no_grad():
                        for data in self.validation_loader:
                            data.pos = data.pos.to(device)
                            data.y = torch.unsqueeze(data.y, 0).to(device)
                            data = data.to(device)
                            outputs = model(data)
                            loss = criterion(outputs, data.y)

                            _, preds = torch.max(outputs, 1)
                            running_loss += loss.detach().item()
                            running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
                            # if i % 20 == 0 and local_rank == 0:
                            #     save_checkpoint(model, optimizer, epoch, val_epoch_acc,f"/data/LIDAR_RA/FSCT/model/ckpt_{epoch}.pth")

                            # i += 1

                        val_epoch_loss = running_loss / len(self.validation_loader)
                        val_epoch_acc = running_acc / len(self.validation_loader)

                        # save_checkpoint(model, optimizer, epoch, val_epoch_acc,
                        #                 f"/data/FSCT/model/ckpt_{epoch}.pth")

                        improved = False
                        if val_epoch_acc > best_val_acc:
                            best_val_acc = val_epoch_acc
                            improved = True

                        if improved:
                            save_checkpoint(model, optimizer, epoch, val_epoch_acc,"/data/FSCT/model/best_model.pth")

                        self.update_log(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc)
                        end_time_train=time.time()

                        print(
                        "Validation epoch accuracy: ", np.around(val_epoch_acc, 4), ", Loss: ", np.around(val_epoch_loss, 4),", Time: ",np.round((end_time_train - start_time_train)/60, 4),
                    )
                        print("=====================================================================")

                del data, outputs, loss, preds
                torch.cuda.empty_cache()

            # torch.save(
            #     model.state_dict(),
            #     os.path.join(get_fsct_path("model"), self.parameters["model_filename"]),
            # )


if __name__ == "__main__":
    parameters = dict(
        preprocess_train_datasets=0,
        preprocess_validation_datasets=0,
        clean_sample_directories=0,  # Deletes all samples in the sample directories.
        perform_validation_during_training=1,
        generate_point_cloud_vis=0,  # Useful for visually checking how well the model is learning. Saves a set of samples called "latest_prediction.las" in the "FSCT/data/"" directory. Samples have label and prediction values.
        load_existing_model=0,
        num_epochs=20000,
        learning_rate=0.000025,
        input_point_cloud=None,
        model_filename_load="best_model.pth",
        sample_box_size_m=np.array([10, 10, 10]),
        sample_box_overlap=[0.65, 0.65, 0.65],
        min_points_per_box=1000,
        max_points_per_box=80000,
        subsample=False,
        subsampling_min_spacing=0.025,
        num_cpu_cores_preprocessing=8,  # 0 Means use all available cores.
        num_cpu_cores_deep_learning=0,  # Setting this higher can cause CUDA issues on Windows.
        train_batch_size=2,
        validation_batch_size=2,
        device="cuda:0",  # set to "cuda" or "cpu"
    )

    mp.set_start_method("spawn", force=True)
    run_training = TrainModel(parameters)
    run_training.run_training()



