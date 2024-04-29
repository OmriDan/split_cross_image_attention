from typing import Tuple, List
import datetime
import nltk
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.cluster import KMeans
from constants import STYLE1_INDEX, STRUCT_INDEX, STYLE2_INDEX
import torch
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

"""
Self-segmentation technique taken from Prompt Mixing: https://github.com/orpatashnik/local-prompt-mixing
"""


class Segmentor:

    def __init__(self, prompt: str, object_nouns: List[str], num_segments: int = 3, res: int = 32):
        self.prompt = prompt
        self.num_segments = num_segments
        self.resolution = res
        self.object_nouns = object_nouns
        print(object_nouns)
        tokenized_prompt = nltk.word_tokenize(prompt)
        forbidden_words = [word.upper() for word in ["photo", "image", "picture"]]
        self.nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_prompt))
                      if pos[:2] == 'NN' and word.upper() not in forbidden_words]

    def update_attention(self, attn, is_cross):
        res = int(attn.shape[2] ** 0.5)
        if is_cross:
            if res == 16:
                self.cross_attention_32 = attn
            elif res == 32:
                self.cross_attention_64 = attn
        else:
            if res == 32:
                self.self_attention_32 = attn
            elif res == 64:
                self.self_attention_64 = attn

    def __call__(self, *args, **kwargs):
        clusters = self.cluster()
        cluster2noun = self.cluster2noun(clusters)
        return cluster2noun


    def visualize_cluster_nouns(self, clusters, noun_assignments, title,step):
        """
        Visualize clusters with annotated nouns.

        Args:
        clusters (np.array): The cluster array where each element is a cluster id.
        noun_assignments (dict): A dictionary where keys are cluster ids and values are the most relevant nouns.
        title (str): The title for the plot.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(clusters, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.axis('off')

        # Annotate each cluster with the corresponding noun
        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            # Find the centroid of the cluster to place the text
            indices = np.where(clusters == cluster_id)
            centroid_x = np.mean(indices[1])
            centroid_y = np.mean(indices[0])
            noun = noun_assignments.get(cluster_id, "BG")
            plt.text(centroid_x, centroid_y, f'{noun[1]}', color='red', ha='center', va='center')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if step % 5 == 0:
            plt.savefig(f"mask_fig/{title.replace(' ', '_').lower()}_{timestamp}_step_{step}.png")
            plt.close()
        # plt.show()

    def cluster(self, res: int = 32):
        np.random.seed(1)
        self_attn = self.self_attention_32 if res == 32 else self.self_attention_64

        style1_attn = self_attn[STYLE1_INDEX].mean(dim=0).cpu().numpy()
        if style1_attn.ndim == 1: # might need to remove
            style1_attn = style1_attn.reshape(-1, 1)
        style1_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(style1_attn)
        style1_clusters = style1_kmeans.labels_.reshape(res, res)

        style2_attn = self_attn[STYLE2_INDEX].mean(dim=0).cpu().numpy()
        if style2_attn.ndim == 1: # might need to remove
            style2_attn = style2_attn.reshape(-1, 1)
        style2_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(style2_attn)
        style2_clusters = style2_kmeans.labels_.reshape(res, res)

        struct_attn = self_attn[STRUCT_INDEX].mean(dim=0).cpu().numpy()
        if struct_attn.ndim == 1: # might need to remove
            struct_attn = struct_attn.reshape(-1, 1)
        struct_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(struct_attn)
        struct_clusters = struct_kmeans.labels_.reshape(res, res)

        return style1_clusters, style2_clusters, struct_clusters

    def visualize_clusters(self, clusters, title,step=1):
        plt.figure(figsize=(8, 8))
        plt.imshow(clusters, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.axis('off')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"mask_fig/{title.replace(' ', '_').lower()}_{timestamp}_step_{step}.png")
        plt.close()
        # plt.show()

    def visualize_masks(self, mask, title, cmap='gray', step=1):
        file_title = f"{title.replace(' ', '_').lower()}"
        with open(f'masks/{file_title}.npy', 'wb') as f:
            np.save(f,mask)
        f.close()
        if step % 5 == 0:
            plt.figure()
            plt.imshow(mask, cmap=cmap)
            plt.colorbar()
            plt.title(title)
            plt.axis('off')
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f"mask_fig/{title.replace(' ', '_').lower()}_{timestamp}_step_{step}.png")
            plt.close()
        # plt.show()

    def cluster2noun(self, clusters, cross_attn, attn_index, is_cross=True):
        result = {}
        res = int(cross_attn.shape[2] ** 0.5)
        nouns_indices = [index for (index, word) in self.nouns]
        cross_attn = cross_attn[attn_index].mean(dim=0).reshape(res, res, -1)
        nouns_maps = cross_attn.cpu().numpy()[:, :, [i + 1 for i in nouns_indices]]
        normalized_nouns_maps = np.zeros_like(nouns_maps).repeat(2, axis=0).repeat(2, axis=1)
        for i in range(nouns_maps.shape[-1]):
            curr_noun_map = nouns_maps[:, :, i].repeat(2, axis=0).repeat(2, axis=1)
            normalized_nouns_maps[:, :, i] = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()

        max_score = 0
        all_scores = []
        for c in range(self.num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            all_scores.append(max(scores))
            max_score = max(max(scores), max_score)

        all_scores.remove(max_score)
        mean_score = sum(all_scores) / len(all_scores)

        for c in range(self.num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            result[c] = self.nouns[np.argmax(np.array(scores))] if max(scores) > 1.4 * mean_score else "BG"

        return result

    def create_mask(self, clusters, cross_attention, attn_index):
        cluster2noun = self.cluster2noun(clusters, cross_attention, attn_index)
        mask = clusters.copy()
        obj_segments = [c for c in cluster2noun if cluster2noun[c][1] in self.object_nouns]
        for c in range(self.num_segments):
            mask[clusters == c] = 1 if c in obj_segments else 0
        return torch.from_numpy(mask).to("cuda")

    def split_structure_mask(self, res=None):
        np.random.seed(1)
        self_attn = self.self_attention_32 if res == 32 else self.self_attention_64
        struct_attn = self_attn[STRUCT_INDEX].mean(dim=0).cpu().numpy()
        if struct_attn.ndim == 1:  # might need to remove
            struct_attn = struct_attn.reshape(-1, 1)
        struct_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(struct_attn)
        struct_clusters = struct_kmeans.labels_.reshape(res, res)

        # Identify the background cluster
        cluster_sizes = np.bincount(struct_clusters.flatten())
        background_cluster = np.argmax(cluster_sizes)

        # Creating binary masks for each cluster
        binary_masks = np.zeros((self.num_segments, res, res))
        for i in range(self.num_segments):
            binary_masks[i] = (struct_clusters == i)

        # Swap to make the first mask the background
        if background_cluster != 0:
            binary_masks[[0, background_cluster]] = binary_masks[[background_cluster, 0]]
        for i in range(self.num_segments):
            binary_masks[i] = torch.from_numpy(binary_masks[i])
        return binary_masks[0], binary_masks[1], binary_masks[2]

    def split_connected_components_and_save(self, mask, name, step, model_self=None, res=None, object_value=1,
                                            save_path='mask_fig/', connectivity=1, use_morphology=True):
        mask_np = mask.cpu().numpy()
        if mask_np.sum() == 0: # mask is completely empty
            if res==32:
                return model_self.object1_mask_32, model_self.object2_mask_32
            else: # res==64
                return model_self.object1_mask_64, model_self.object2_mask_64

        if use_morphology:
            # Initial settings for morphological operations
            erosion_size = 3  # Initial erosion size
            dilation_size = 2  # Initial dilation size

            while True:
                # Apply morphological erosion and dilation
                eroded_mask = binary_erosion(mask_np == object_value, structure=np.ones((erosion_size, erosion_size)))
                processed_mask = binary_dilation(eroded_mask, structure=np.ones((dilation_size, dilation_size)))

                # Find connected components
                labeled_mask, num_features = label(processed_mask,
                                                   structure=np.ones((3, 3)) if connectivity == 8 else None)

                if num_features >= 2:
                    break  # Exit the loop if successful
                else:
                    # Increase erosion and decrease dilation size to attempt further separation
                    erosion_size += 1
                    dilation_size = max(1, dilation_size - 1)
                    if erosion_size > 6:  # Avoid excessively large erosion which might remove all features
                        #print("Unable to separate components with morphological operations.")
                        if res == 32:
                            return model_self.object1_mask_32, model_self.object2_mask_32
                        else:  # res==64
                            return model_self.object1_mask_64, model_self.object2_mask_64
        else:
            processed_mask = mask_np == object_value
            labeled_mask, num_features = label(processed_mask, structure=np.ones((3, 3)) if connectivity == 8 else None)
            if num_features < 2:
                print("Less than two separate objects found.")
                if res == 32:
                    return model_self.object1_mask_32, model_self.object2_mask_32
                else:  # res==64
                    return model_self.object1_mask_64, model_self.object2_mask_64
                    # Proceed with saving the separated components
        fig, axes = plt.subplots(1, min(num_features, 2), figsize=(12, 6))
        masks = []
        for i in range(1, min(num_features, 2) + 1):
            mask_i = torch.from_numpy((labeled_mask == i).astype(int))
            masks.append(mask_i)
            ax = axes[i - 1] if num_features > 1 else axes
            ax.imshow(mask_i.cpu().numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Object {i} Mask')
            if i == 2 and step % 5 == 0:
                plt.savefig(f"{save_path}{name}_{step}_object{i}.png")
        plt.close(fig)
        return tuple(masks)
        
    def get_object_masks(self, is_cross=True, step=1, use_cluster = True) -> Tuple[torch.Tensor]:
        mask_style1_32, mask_style2_32, mask_struct_32, mask_style1_64, mask_style2_64, mask_struct_64 = tuple(torch.empty(0) for _ in range(6))
        flag_32 = hasattr(self, 'self_attention_32') or hasattr(self, 'cross_attention_32')
        flag_64 = hasattr(self, 'self_attention_64') or hasattr(self, 'cross_attention_64')
        if is_cross:
            title_addition = 'cross_attention'
            if flag_32:
                attn_32 = self.cross_attention_32
            if flag_64:
                attn_64 = self.cross_attention_64
        else:
            title_addition = 'self_attention'
            if flag_32:
                attn_32 = self.self_attention_32
            if flag_64:
                attn_64 = self.self_attention_64
        if flag_32:
            # Get cluster to noun mappings for visualization
            clusters_style1_32, clusters_style2_32, clusters_struct_32 = self.cluster(res=32)
            if use_cluster:
                cluster2noun_32_style1 = self.cluster2noun(clusters_style1_32, attn_32, STYLE1_INDEX)
                cluster2noun_32_style2 = self.cluster2noun(clusters_style2_32, attn_32, STYLE2_INDEX)
                cluster2noun_32_struct = self.cluster2noun(clusters_struct_32, attn_32, STRUCT_INDEX)
                # Visualize clusters with nouns
                self.visualize_cluster_nouns(clusters_style1_32, cluster2noun_32_style1,
                                             f'{title_addition} Style1 Clusters Resolution 32 with Nouns step {step}',step=step)
                self.visualize_cluster_nouns(clusters_style2_32, cluster2noun_32_style2,
                                             f'{title_addition} Style2 Clusters Resolution 32 with Nouns step {step}', step=step)
                self.visualize_cluster_nouns(clusters_struct_32, cluster2noun_32_struct,
                                             f'{title_addition} Structural Clusters Resolution 32 with Nouns step {step}',step=step)

            mask_style1_32 = self.create_mask(clusters_style1_32, attn_32, STYLE1_INDEX)
            mask_style2_32 = self.create_mask(clusters_style2_32, attn_32, STYLE2_INDEX)
            mask_struct_32 = self.create_mask(clusters_struct_32, attn_32, STRUCT_INDEX)

            # Visualizing and saving the masks
            self.visualize_masks(mask_style1_32.cpu().numpy(), f'{title_addition} Style1 Mask Resolution 32 step {step}',step=step)
            self.visualize_masks(mask_style2_32.cpu().numpy(), f'{title_addition} Style2 Mask Resolution 32',step=step)
            self.visualize_masks(mask_struct_32.cpu().numpy(), f'{title_addition} Structural Mask Resolution 32 step {step}', step=step)
        if flag_64:
            # Get cluster to noun mappings for visualization
            clusters_style1_64, clusters_style2_64, clusters_struct_64 = self.cluster(res=64)
            if use_cluster:
                cluster2noun_64_style1 = self.cluster2noun(clusters_style1_64, attn_64, STYLE1_INDEX)
                cluster2noun_64_style2 = self.cluster2noun(clusters_style2_64, attn_64, STYLE2_INDEX)
                cluster2noun_64_struct = self.cluster2noun(clusters_struct_64, attn_64, STRUCT_INDEX)
                # Visualize clusters with nouns
                self.visualize_cluster_nouns(clusters_style1_64, cluster2noun_64_style1,
                                             f'{title_addition} Style1 Clusters Resolution 64 with Nouns',step=step)
                self.visualize_cluster_nouns(clusters_style2_64, cluster2noun_64_style2,
                                             f'{title_addition} Style2 Clusters Resolution 64 with Nouns', step=step)
                self.visualize_cluster_nouns(clusters_struct_64, cluster2noun_64_struct,
                                             f'{title_addition} Structural Clusters Resolution 64 with Nouns', step=step)

            mask_style1_64 = self.create_mask(clusters_style1_64, attn_64, STYLE1_INDEX)
            mask_style2_64 = self.create_mask(clusters_style2_64, attn_64, STYLE1_INDEX)
            mask_struct_64 = self.create_mask(clusters_struct_64, attn_64, STRUCT_INDEX)

            # Visualizing and saving the masks
            self.visualize_masks(mask_style1_64.cpu().numpy(), f'{title_addition} Style1 Mask Resolution 64', step=step)
            self.visualize_masks(mask_style2_64.cpu().numpy(), f'{title_addition} Style2 Mask Resolution 64',step=step)
            self.visualize_masks(mask_struct_64.cpu().numpy(), f'{title_addition} Structural Mask Resolution 64',step=step)
        # Add visualization for clusters
        # self.visualize_clusters(clusters_style1_32, 'Style1 Clusters Resolution 32')
        # self.visualize_clusters(clusters_style2_32, 'Style2 Clusters Resolution 32')
        # self.visualize_clusters(clusters_struct_32, 'Structural Clusters Resolution 32')
        # self.visualize_clusters(clusters_style1_64, 'Style1 Clusters Resolution 64')
        # self.visualize_clusters(clusters_style2_64, 'Style2 Clusters Resolution 64')
        # self.visualize_clusters(clusters_struct_64, 'Structural Clusters Resolution 64')
        return mask_style1_32, mask_style2_32, mask_struct_32, mask_style1_64, mask_style2_64, mask_struct_64
